"""Lazy finite-grid compilation and artifact collection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..artifacts.io import sha256_path
from ..artifacts.manifest import new_grid_manifest, write_manifest_atomic
from ..artifacts.store import RunStore
from ..artifacts.tables import MetricProjection
from ..evaluation.plans import ScenarioKey
from ..evaluation.selected import SelectedEvaluator, materialize_selected_evaluator
from ..specs.grid import GridSpec
from ..specs.pair import PairSpec
from ..specs.scenario import ScenarioClosure, ScenarioSpec
from .model import CartesianGridPlan, GridValidationError, compile_grid_plan


@dataclass(frozen=True)
class GridCompilation:
    """Immutable axes-only request compilation."""

    run_dir: Path
    manifest_path: Path
    plan: CartesianGridPlan
    manifest: Mapping[str, Any]


def _artifact(path: Path, root: Path, kind: str) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "kind": kind,
        "bytes": path.stat().st_size,
        "sha256": sha256_path(path),
    }


def _validate_specs(
    grid_spec: GridSpec,
    pair_spec: PairSpec,
    scenario_spec: ScenarioSpec,
    metric_projection: MetricProjection,
) -> None:
    if pair_spec.id != grid_spec.pair_id:
        raise ValueError(f"grid pair {grid_spec.pair_id!r} does not match pair spec {pair_spec.id!r}")
    if scenario_spec.pair_id != pair_spec.id:
        raise ValueError(f"scenario pair {scenario_spec.pair_id!r} does not match {pair_spec.id!r}")
    if not metric_projection.fields:
        raise ValueError("grid execution requires a non-empty MetricProjection")


def compile_grid_run(
    grid_spec: GridSpec,
    *,
    run_id: str,
    pair_spec: PairSpec,
    scenario_spec: ScenarioSpec,
    store: RunStore,
    metric_projection: MetricProjection,
    selected_evaluator: SelectedEvaluator,
    open_session: Mapping[str, object],
    scenario: ScenarioClosure | ScenarioKey,
) -> GridCompilation:
    """Persist an axes-only plan without compiling any Cartesian point."""
    _validate_specs(grid_spec, pair_spec, scenario_spec, metric_projection)
    if not isinstance(selected_evaluator, SelectedEvaluator):
        raise TypeError("selected_evaluator must be a SelectedEvaluator")
    compiler = selected_evaluator.compiler
    core = selected_evaluator.manifest_core()
    provenance = selected_evaluator.provenance
    if selected_evaluator.parameter_schema_sha256 != compiler.schema.sha256:
        raise ValueError("selected evaluator compiler schema does not match its artifact")
    if grid_spec.policy_id is not None and grid_spec.policy_id != compiler.schema.policy_id:
        raise ValueError("grid policy does not match selected evaluator schema")
    if provenance.get("binary_sha256") != core.get("sha256") or provenance.get(
        "parameter_schema_sha256"
    ) != compiler.schema.sha256:
        raise ValueError("selected evaluator provenance does not match its manifest core")

    snap_receipt: dict[str, Any] = {}
    plan = compile_grid_plan(
        grid_spec,
        compiler=compiler,
        artifact_sha256=selected_evaluator.artifact_sha256,
        open_session=open_session,
        scenario=scenario,
        snap_receipt=snap_receipt,
    )
    policy = {
        "id": compiler.schema.policy_id,
        "source_sha256": core.get("policy_source_sha256"),
        "policy_abi": core.get("policy_abi"),
        "parameter_names": [
            descriptor.name
            for descriptor in sorted(
                (item for item in compiler.schema.descriptors if item.order is not None),
                key=lambda item: item.order,
            )
        ],
    }
    resolved_spec = {
        "grid": {
            "id": grid_spec.id,
            "pair_id": grid_spec.pair_id,
            "policy_id": grid_spec.policy_id,
            "tags": list(grid_spec.tags),
        },
        "pair": pair_spec.to_dict(),
        "scenario": scenario_spec.to_dict(),
        "policy": policy,
        "metric_projection": metric_projection.to_dict(),
        "grid_snap_receipt": snap_receipt,
        "evaluator_artifact_selection": provenance,
    }

    run_dir = store.allocate_run_dir("grid", run_id)
    selected_evaluator = materialize_selected_evaluator(
        selected_evaluator, run_dir, resume=False
    )
    if selected_evaluator.provenance != provenance:
        raise ValueError("run-local evaluator selection differs from compiled grid selection")
    core = selected_evaluator.manifest_core(binary_override="evaluator_artifact/evaluator")
    artifacts = tuple(
        _artifact(run_dir / "evaluator_artifact" / name, run_dir, kind)
        for name, kind in (
            ("artifact.json", "evaluator_artifact_receipt"),
            ("evaluator", "evaluator_binary"),
        )
    )
    manifest = new_grid_manifest(
        run_id=run_id,
        grid_id=grid_spec.id,
        pool_count=plan.pool_count,
        resolved_spec=resolved_spec,
        plan=plan.to_dict(),
        core=core,
        artifacts=artifacts,
    )
    manifest_path = run_dir / "manifest.json"
    write_manifest_atomic(manifest_path, manifest, expected_kind="grid")
    return GridCompilation(run_dir, manifest_path, plan, manifest)


def load_grid_plan(
    manifest: Mapping[str, Any],
    *,
    selected_evaluator: SelectedEvaluator,
    scenario: ScenarioClosure | ScenarioKey,
) -> CartesianGridPlan:
    """Load and verify the canonical axes-only plan from a v2 manifest."""
    if not isinstance(selected_evaluator, SelectedEvaluator):
        raise TypeError("selected_evaluator must be a SelectedEvaluator")
    grid = manifest.get("grid")
    if not isinstance(grid, Mapping) or not isinstance(grid.get("plan"), Mapping):
        raise GridValidationError("grid manifest has no Cartesian plan")
    try:
        plan = CartesianGridPlan.from_dict(
            grid["plan"],
            compiler=selected_evaluator.compiler,
            artifact_sha256=selected_evaluator.artifact_sha256,
            scenario=scenario,
        )
    except Exception as exc:
        raise GridValidationError(f"invalid Cartesian grid plan: {exc}") from exc
    if grid.get("grid_id") != plan.grid_id or grid.get("pool_count") != plan.pool_count:
        raise GridValidationError("grid branch differs from its Cartesian plan")
    return plan


__all__ = [
    "GridCompilation",
    "compile_grid_plan",
    "compile_grid_run",
    "load_grid_plan",
]
