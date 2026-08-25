"""Replay one grid point with full evaluator tracing."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

from .contract import Candidate
from .placement import EvaluatorFleet, PlacementLane, local_client_factory
from .engine import ClientFactory
from .run import (
    RunConfig,
    open_session_request,
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
    """Replay one explicitly stored candidate through one local evaluator."""
    if not isinstance(candidate, Candidate):
        raise TypeError("candidate must be a Candidate")
    if ordinal < 0:
        raise ValueError("ordinal must be non-negative")
    if trace_interval < 1:
        raise ValueError("trace_interval must be positive")
    config = RunConfig.from_toml(config_path)
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    open_session = open_session_request(config, remote=False)
    if yb_mode is not None:
        if yb_mode not in {"off", "passive", "active_2l"}:
            raise ValueError("yb_mode must be off, passive, or active_2l")
        open_session["yb_mode"] = yb_mode
    factory = client_factory or local_client_factory(
        config.evaluator, work_dir=config.path.parent, workers=1
    )
    with EvaluatorFleet(
        (PlacementLane("injected" if client_factory else "local", factory),),
        session_id=f"{config.run_id}-trace-{ordinal}",
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
        "run_id": config.run_id,
        "source_ordinal": ordinal,
        "candidate": candidate.to_dict(ordinal=ordinal),
        "result": result.to_dict(),
    })
    return summary


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


__all__ = ["save_shiftclick_plot", "shiftclick_figure", "trace_candidate"]
