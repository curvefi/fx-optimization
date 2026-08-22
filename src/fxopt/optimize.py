"""Nevergrad optimization over the same evaluator fleet used by grids."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import tomllib
from typing import Any

import nevergrad as ng

from .candidates import CandidateSpec, merge_payload
from .placement import EvaluatorFleet
from .engine import ClientFactory
from .results import ArtifactPaths, ResultWriter
from .run import (
    ProgressCallback,
    RunConfig,
    candidate_from_spec,
    open_session_request,
    placement_lanes,
    run_metadata,
)


class OptimizationError(ValueError):
    pass


def _settings(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        raw = tomllib.load(stream).get("optimization")
    if not isinstance(raw, Mapping):
        raise OptimizationError("config requires an [optimization] table")
    allowed = {"budget", "batch_size", "metric", "maximize", "seed"}
    unknown = set(raw) - allowed
    if unknown:
        raise OptimizationError(f"unknown [optimization] keys: {sorted(unknown)}")
    budget = raw.get("budget")
    batch_size = raw.get("batch_size", 1)
    metric = raw.get("metric")
    maximize = raw.get("maximize", True)
    seed = raw.get("seed", 0)
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise OptimizationError("optimization.budget must be a positive integer")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise OptimizationError("optimization.batch_size must be a positive integer")
    if not isinstance(metric, str) or not metric:
        raise OptimizationError("optimization.metric must be a non-empty string")
    if not isinstance(maximize, bool):
        raise OptimizationError("optimization.maximize must be a boolean")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise OptimizationError("optimization.seed must be an integer")
    return {"budget": budget, "batch_size": batch_size, "metric": metric,
            "maximize": maximize, "seed": seed}


def optimize_config(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    client_factory: ClientFactory | None = None,
    progress_callback: ProgressCallback | None = None,
) -> ArtifactPaths:
    """Optimize discrete candidate axes locally or over configured SSH lanes."""
    config = RunConfig.from_toml(config_path)
    settings = _settings(config.path)
    if not config.candidate.axes:
        raise OptimizationError("optimization requires at least one candidate axis")
    parametrization = ng.p.Dict(**{
        name: ng.p.Choice(list(values)) for name, values in config.candidate.axes.items()
    })
    optimizer = ng.optimizers.TwoPointsDE(
        parametrization=parametrization,
        budget=settings["budget"],
        num_workers=settings["batch_size"],
    )
    optimizer.parametrization.random_state.seed(settings["seed"])

    destination = Path(output_dir).expanduser().resolve()
    lanes = placement_lanes(config, client_factory)
    per_lane_batch = min(config.batch_size, settings["batch_size"])
    metadata = run_metadata(config, effective_batch=per_lane_batch)
    metadata.update({"kind": "optimization", **settings})
    writer = ResultWriter(destination, run_id=config.run_id, metadata=metadata)
    best_value = float("-inf") if settings["maximize"] else float("inf")
    best_ordinal: int | None = None
    completed = 0

    with writer:
        total = settings["budget"]
        if progress_callback is not None:
            progress_callback(0, total)
        with EvaluatorFleet(
            lanes,
            session_id=config.run_id,
            batch_size=per_lane_batch,
            open_session=open_session_request(config),
        ) as fleet:
            while completed < settings["budget"]:
                count = min(settings["batch_size"], settings["budget"] - completed)
                asked = [optimizer.ask() for _ in range(count)]
                specs = [
                    CandidateSpec.from_payload(
                        merge_payload(config.candidate.defaults, item.value),
                        ordinal=completed + index,
                    )
                    for index, item in enumerate(asked)
                ]
                candidates = tuple(candidate_from_spec(spec) for spec in specs)
                results = fleet.evaluate(candidates)
                for index, (asked_item, result) in enumerate(zip(asked, results, strict=True)):
                    value = result.metrics.get(settings["metric"])
                    valid = result.status == "ok" and value is not None
                    loss = (-value if settings["maximize"] else value) if valid else 1e300
                    optimizer.tell(asked_item, loss)
                    if valid and ((settings["maximize"] and value > best_value) or
                                  (not settings["maximize"] and value < best_value)):
                        best_value = value
                        best_ordinal = completed + index
                writer.append(candidates, results)
                completed += count
                if progress_callback is not None:
                    progress_callback(completed, total)
        writer.update_metadata(best_ordinal=best_ordinal,
                               best_metric_value=None if best_ordinal is None else best_value)
        return writer.finalize()


__all__ = ["OptimizationError", "optimize_config"]
