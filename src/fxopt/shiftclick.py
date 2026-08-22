"""Replay one grid point with full evaluator tracing."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

from .contract import Candidate
from .placement import EvaluatorFleet
from .engine import ClientFactory
from .run import (
    RunConfig,
    open_session_request,
    placement_lanes,
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
    """Replay one explicitly stored candidate through the source run's fleet."""
    if not isinstance(candidate, Candidate):
        raise TypeError("candidate must be a Candidate")
    if ordinal < 0:
        raise ValueError("ordinal must be non-negative")
    if trace_interval < 1:
        raise ValueError("trace_interval must be positive")
    config = RunConfig.from_toml(config_path)
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    open_session = open_session_request(config)
    if yb_mode is not None:
        if yb_mode not in {"off", "passive", "active_2l"}:
            raise ValueError("yb_mode must be off, passive, or active_2l")
        open_session["yb_mode"] = yb_mode
    with EvaluatorFleet(
        placement_lanes(config, client_factory),
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


__all__ = ["trace_candidate"]
