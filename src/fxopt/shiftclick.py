"""Replay one grid point with full evaluator tracing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from .contract import Candidate
from .engine import ClientFactory, EvaluatorSession
from .placement import local_client_factory
from .run import EVALUATOR_POLICY_METADATA_KEY


@dataclass(frozen=True, slots=True)
class ReplaySpec:
    """Local evaluator and session inputs embedded in a completed run."""

    run_id: str
    evaluator: Path
    work_dir: Path
    open_session: Mapping[str, Any]
    evaluator_policy: Mapping[str, str | int]


def _replay_from_metadata(
    run_id: str,
    metadata: Mapping[str, Any],
) -> ReplaySpec:
    raw = metadata.get("replay")
    if not isinstance(raw, Mapping):
        raise ValueError("run metadata has no self-contained replay inputs")
    evaluator = raw.get("evaluator")
    work_dir = raw.get("work_dir")
    open_session = raw.get("open_session")
    evaluator_policy = metadata.get(EVALUATOR_POLICY_METADATA_KEY)
    if not isinstance(evaluator, str) or not evaluator:
        raise ValueError("run replay metadata has no local evaluator")
    if not isinstance(work_dir, str) or not work_dir:
        raise ValueError("run replay metadata has no local work directory")
    if not isinstance(open_session, Mapping):
        raise ValueError("run replay metadata has no local session request")
    normalized_open_session = dict(open_session)
    legacy_arbitrage = normalized_open_session.pop("arbitrage_enabled", True)
    if legacy_arbitrage is not True:
        raise ValueError(
            "cannot replay a historical no-arbitrage run after removal of "
            "arbitrage_enabled"
        )
    if (
        not isinstance(evaluator_policy, Mapping)
        or set(evaluator_policy)
        != {"policy_id", "policy_abi", "policy_parameter_count"}
        or any(
            not isinstance(evaluator_policy[name], str)
            or not evaluator_policy[name]
            for name in ("policy_id", "policy_abi")
        )
        or isinstance(evaluator_policy["policy_parameter_count"], bool)
        or not isinstance(evaluator_policy["policy_parameter_count"], int)
        or evaluator_policy["policy_parameter_count"] < 0
    ):
        raise ValueError("run metadata has no valid expected evaluator policy")
    return ReplaySpec(
        run_id=run_id,
        evaluator=Path(evaluator).expanduser(),
        work_dir=Path(work_dir).expanduser(),
        open_session=normalized_open_session,
        evaluator_policy=dict(evaluator_policy),
    )


def _atomic_json(path: Path, value: object) -> None:
    handle = tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _run_trace(
    replay: ReplaySpec,
    *,
    candidate: Candidate,
    ordinal: int,
    output_dir: str | Path,
    trace_interval: int = 1,
    trace_actions: bool = False,
    client_factory: ClientFactory | None = None,
    yb_mode: str | None = None,
    yb_cash_multiplier: float | None = None,
) -> Path:
    if not isinstance(candidate, Candidate):
        raise TypeError("candidate must be a Candidate")
    if ordinal < 0:
        raise ValueError("ordinal must be non-negative")
    if trace_interval < 1:
        raise ValueError("trace_interval must be positive")
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    open_session = dict(replay.open_session)
    open_session["event_cursor"] = "scalar"
    open_session["metric_profile"] = "full_summary"
    if yb_mode is not None:
        if yb_mode not in {"off", "active_2l", "reference_2l"}:
            raise ValueError(
                "yb_mode must be off, active_2l, or reference_2l"
            )
        open_session["yb_mode"] = yb_mode
    if yb_cash_multiplier is not None:
        if not math.isfinite(yb_cash_multiplier) or yb_cash_multiplier <= 0.0:
            raise ValueError("yb_cash_multiplier must be finite and positive")
        open_session["yb_cash_multiplier"] = yb_cash_multiplier
    factory = client_factory or local_client_factory(
        replay.evaluator,
        work_dir=replay.work_dir,
        workers=1,
        client_options={
            f"expected_{name}": value
            for name, value in replay.evaluator_policy.items()
        },
    )
    with EvaluatorSession(
        factory,
        session_id=f"{replay.run_id}-trace-{ordinal}",
        open_session=open_session,
        metric_projection="full",
        observation={
            "kind": "full_trace",
            "trace_interval": trace_interval,
            "trace_actions": trace_actions,
            "artifact_dir": str(destination),
        },
    ) as session:
        result = session.evaluate((candidate,))[0]

    summary = destination / "shiftclick.json"
    _atomic_json(summary, {
        "run_id": replay.run_id,
        "source_ordinal": ordinal,
        "candidate": candidate.to_dict(ordinal=ordinal),
        "result": result.to_dict(),
    })
    return summary


def trace_stored_candidate(
    run_id: str,
    metadata: Mapping[str, Any],
    *,
    candidate: Candidate,
    ordinal: int,
    output_dir: str | Path,
    trace_interval: int = 1,
    trace_actions: bool = False,
    client_factory: ClientFactory | None = None,
    yb_mode: str | None = None,
    yb_cash_multiplier: float | None = None,
) -> Path:
    """Replay from self-contained run metadata."""
    return _run_trace(
        _replay_from_metadata(run_id, metadata),
        candidate=candidate,
        ordinal=ordinal,
        output_dir=output_dir,
        trace_interval=trace_interval,
        trace_actions=trace_actions,
        client_factory=client_factory,
        yb_mode=yb_mode,
        yb_cash_multiplier=yb_cash_multiplier,
    )


def shiftclick_figure(summary_path: str | Path, *, title: str | None = None):
    """Render the trace referenced by a Shift-click summary."""
    summary = Path(summary_path).expanduser().resolve()
    try:
        payload = json.loads(summary.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid Shift-click summary: {summary}") from exc
    if (
        not isinstance(payload, Mapping)
        or not isinstance(payload.get("result"), Mapping)
    ):
        raise ValueError(f"invalid Shift-click summary: {summary}")
    artifacts = payload["result"].get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError(f"Shift-click summary has no artifacts: {summary}")
    raw_trace = artifacts.get("trace_path")
    effective_inputs = artifacts.get("effective_inputs")
    if not isinstance(effective_inputs, Mapping):
        raise ValueError(
            f"Shift-click summary has no effective_inputs; cannot compute net APY: {summary}"
        )
    if (
        "pool.donation_frequency" not in effective_inputs
        or effective_inputs["pool.donation_frequency"] is None
    ):
        raise ValueError(
            "Shift-click effective_inputs.pool.donation_frequency must be a finite number"
        )
    donation_frequency = effective_inputs["pool.donation_frequency"]
    if (
        isinstance(donation_frequency, bool)
        or not isinstance(donation_frequency, (int, float))
    ):
        raise ValueError(
            "Shift-click effective_inputs.pool.donation_frequency must be a finite number"
        )
    donation_frequency = float(donation_frequency)
    if not math.isfinite(donation_frequency) or donation_frequency < 0.0:
        raise ValueError(
            "Shift-click effective_inputs.pool.donation_frequency must be a "
            "finite non-negative number"
        )
    if not isinstance(raw_trace, str) or not raw_trace:
        raise ValueError(f"Shift-click summary has no trace path: {summary}")
    trace = Path(raw_trace).expanduser()
    if not trace.is_absolute():
        trace = summary.parent / trace
    if not trace.is_file():
        raise ValueError(f"Shift-click trace does not exist locally: {trace}")
    from curve_fx_sim.plotting.shiftclick_view import render_shiftclick_figure

    return render_shiftclick_figure(
        trace,
        title=title,
        donation_frequency=donation_frequency,
    )


def save_shiftclick_plot(
    summary_path: str | Path,
    output_path: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Render and atomically save one Shift-click PNG."""
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    figure = shiftclick_figure(summary_path, title=title)
    try:
        figure.savefig(temporary, format="png", dpi=160, bbox_inches="tight")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
        from matplotlib import pyplot as plt

        plt.close(figure)
    return output


__all__ = [
    "ReplaySpec",
    "save_shiftclick_plot",
    "shiftclick_figure",
    "trace_stored_candidate",
]
