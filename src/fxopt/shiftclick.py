"""Replay one grid point with full evaluator tracing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from .contract import Candidate
from .placement import EvaluatorFleet, PlacementLane, local_client_factory
from .engine import ClientFactory
from .run import (
    RunConfig,
    open_session_request,
)


@dataclass(frozen=True, slots=True)
class ReplaySpec:
    """Local evaluator and session inputs embedded in a completed run."""

    run_id: str
    evaluator: Path
    work_dir: Path
    open_session: Mapping[str, Any]


def _replay_from_config(config: RunConfig) -> ReplaySpec:
    return ReplaySpec(
        run_id=config.run_id,
        evaluator=config.evaluator,
        work_dir=config.path.parent,
        open_session=open_session_request(config, remote=False),
    )


def _replay_from_metadata(
    run_id: str,
    metadata: Mapping[str, Any],
) -> ReplaySpec:
    raw = metadata.get("replay")
    if not isinstance(raw, Mapping):
        config_path = metadata.get("config")
        if not isinstance(config_path, str):
            raise ValueError("run metadata has neither replay inputs nor a source config")
        return _replay_from_config(RunConfig.from_toml(config_path))
    evaluator = raw.get("evaluator")
    work_dir = raw.get("work_dir")
    open_session = raw.get("open_session")
    if not isinstance(evaluator, str) or not evaluator:
        raise ValueError("run replay metadata has no local evaluator")
    if not isinstance(work_dir, str) or not work_dir:
        raise ValueError("run replay metadata has no local work directory")
    if not isinstance(open_session, Mapping):
        raise ValueError("run replay metadata has no local session request")
    return ReplaySpec(
        run_id=run_id,
        evaluator=Path(evaluator).expanduser(),
        work_dir=Path(work_dir).expanduser(),
        open_session=dict(open_session),
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


def _trace_candidate(
    replay: ReplaySpec,
    *,
    candidate: Candidate,
    ordinal: int,
    output_dir: str | Path,
    trace_interval: int = 1,
    trace_actions: bool = False,
    client_factory: ClientFactory | None = None,
    yb_mode: str | None = None,
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
    if yb_mode is not None:
        if yb_mode not in {"off", "passive", "active_2l"}:
            raise ValueError("yb_mode must be off, passive, or active_2l")
        open_session["yb_mode"] = yb_mode
    factory = client_factory or local_client_factory(
        replay.evaluator, work_dir=replay.work_dir, workers=1
    )
    with EvaluatorFleet(
        (PlacementLane("injected" if client_factory else "local", factory),),
        session_id=f"{replay.run_id}-trace-{ordinal}",
        batch_size=1,
        open_session=open_session,
        metric_projection="full",
        observation={
            "kind": "full_trace",
            "trace_interval": trace_interval,
            "trace_actions": trace_actions,
            "artifact_dir": str(destination),
        },
    ) as fleet:
        result = fleet.evaluate((candidate,))[0]

    summary = destination / "shiftclick.json"
    _atomic_json(summary, {
        "run_id": replay.run_id,
        "source_ordinal": ordinal,
        "candidate": candidate.to_dict(ordinal=ordinal),
        "result": result.to_dict(),
    })
    return summary


def trace_candidate(
    config_path: str | Path,
    *,
    candidate: Candidate,
    ordinal: int,
    output_dir: str | Path,
    trace_interval: int = 1,
    trace_actions: bool = False,
    client_factory: ClientFactory | None = None,
    yb_mode: str | None = None,
) -> Path:
    """Replay a candidate from a source config (legacy/public API)."""
    return _trace_candidate(
        _replay_from_config(RunConfig.from_toml(config_path)),
        candidate=candidate,
        ordinal=ordinal,
        output_dir=output_dir,
        trace_interval=trace_interval,
        trace_actions=trace_actions,
        client_factory=client_factory,
        yb_mode=yb_mode,
    )


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
) -> Path:
    """Replay from self-contained run metadata, with source-config fallback."""
    return _trace_candidate(
        _replay_from_metadata(run_id, metadata),
        candidate=candidate,
        ordinal=ordinal,
        output_dir=output_dir,
        trace_interval=trace_interval,
        trace_actions=trace_actions,
        client_factory=client_factory,
        yb_mode=yb_mode,
    )


def shiftclick_figure(summary_path: str | Path, *, title: str | None = None):
    """Render the trace referenced by a Shift-click summary."""
    summary = Path(summary_path).expanduser().resolve()
    try:
        payload = json.loads(summary.read_text(encoding="utf-8"))
        raw_trace = payload["result"]["artifacts"]["trace_path"]
    except (OSError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid Shift-click summary: {summary}") from exc
    if not isinstance(raw_trace, str) or not raw_trace:
        raise ValueError(f"Shift-click summary has no trace path: {summary}")
    trace = Path(raw_trace).expanduser()
    if not trace.is_absolute():
        trace = summary.parent / trace
    if not trace.is_file():
        raise ValueError(f"Shift-click trace does not exist locally: {trace}")
    from curve_fx_sim.plotting.shiftclick_view import render_shiftclick_figure

    return render_shiftclick_figure(trace, title=title)


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
    "trace_candidate",
    "trace_stored_candidate",
]
