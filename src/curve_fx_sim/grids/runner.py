"""Finite-grid compilation and shared artifact collection."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from curve_fx_harness_client.models import CandidateResult

from ..artifacts.io import sha256_path
from ..execution.collection import load_grid_results_npz
from ..artifacts.manifest import load_manifest, new_grid_manifest, write_manifest_atomic
from ..artifacts.store import RunStore
from ..artifacts.tables import EvaluationTable, MetricProjection
from ..evaluation.identity import VerifiedEvaluator, validate_evaluator_identity
from ..specs.grid import GridSpec
from ..specs.pair import PairSpec
from ..specs.policy import PolicySpec
from ..specs.scenario import ScenarioSpec
from ..optimization.profiles import profile_from_policy_spec, quantized
from .collection import GridCoverageError, collect_evaluations
from .model import GridPoint, coordinate_signature, expand_grid


@dataclass(frozen=True)
class GridCompilation:
    """Immutable request compilation consumed by local and cluster execution."""

    run_dir: Path
    manifest_path: Path
    points: tuple[GridPoint, ...]
    manifest: Mapping[str, Any]


@dataclass(frozen=True)
class GridRunResult:
    run_dir: Path
    points: tuple[GridPoint, ...]
    table: EvaluationTable
    manifest: Mapping[str, Any]


def _artifact(path: Path, root: Path, kind: str) -> dict[str, Any]:
    return {"path": path.relative_to(root).as_posix(), "kind": kind, "bytes": path.stat().st_size, "sha256": sha256_path(path)}


def _core_identity(identity: VerifiedEvaluator | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(identity, VerifiedEvaluator):
        return identity.to_core_dict()
    core = copy.deepcopy(dict(identity))
    if "binary" not in core and "path" in core:
        core["binary"] = core["path"]
    if "sha256" not in core and "binary_sha256" in core:
        core["sha256"] = core["binary_sha256"]
    if not core.get("sha256"):
        raise ValueError("grid compilation requires an attested evaluator SHA-256")
    return core


def _validate_specs(
    grid_spec: GridSpec,
    pair_spec: PairSpec,
    scenario_spec: ScenarioSpec,
    policy_spec: PolicySpec | None,
    metric_projection: MetricProjection,
) -> None:
    if pair_spec.id != grid_spec.pair_id:
        raise ValueError(f"grid pair {grid_spec.pair_id!r} does not match pair spec {pair_spec.id!r}")
    if scenario_spec.pair_id != pair_spec.id:
        raise ValueError(f"scenario pair {scenario_spec.pair_id!r} does not match {pair_spec.id!r}")
    if not metric_projection.fields:
        raise ValueError("grid execution requires a non-empty MetricProjection")
    if policy_spec is None:
        if grid_spec.policy_id is not None:
            raise ValueError(
                f"grid policy {grid_spec.policy_id!r} requires a compiled policy specification"
            )
        return
    if grid_spec.policy_id != policy_spec.id:
        raise ValueError(
            f"grid policy {grid_spec.policy_id!r} does not match policy spec {policy_spec.id!r}"
        )
    if policy_spec.policy_kind != "compiled" or not policy_spec.source_sha256:
        raise ValueError("compiled-policy grid requires one source-attested policy")


def compile_grid_run(
    grid_spec: GridSpec,
    *,
    run_id: str,
    pair_spec: PairSpec,
    scenario_spec: ScenarioSpec,
    policy_spec: PolicySpec | None,
    store: RunStore,
    metric_projection: MetricProjection,
    evaluator_identity: VerifiedEvaluator | Mapping[str, Any],
) -> GridCompilation:
    """Compile one finite native-pool or compiled-policy grid."""
    _validate_specs(grid_spec, pair_spec, scenario_spec, policy_spec, metric_projection)

    if policy_spec is None:
        expected_identity = (
            ("policy_id", "twocrypto_native"),
            ("policy_source_sha256", "none"),
            ("policy_parameter_count", 0),
        )
        policy_dict: dict[str, Any] = {
            "id": "twocrypto_native",
            "policy_kind": "native",
            "parameters": [],
        }
    else:
        header_path = (store.root_dir / policy_spec.header_file).resolve()
        if not header_path.is_file():
            raise FileNotFoundError(f"compiled policy header not found: {header_path}")
        header_sha256 = sha256_path(header_path)
        if header_sha256 != policy_spec.source_sha256:
            raise ValueError(
                f"policy source SHA-256 mismatch: spec {policy_spec.source_sha256}, file {header_sha256}"
            )
        expected_identity = (
            ("policy_id", policy_spec.id),
            ("policy_source_sha256", policy_spec.source_sha256),
            ("policy_abi", policy_spec.policy_abi),
            ("policy_parameter_count", len(policy_spec.parameters)),
        )
        policy_dict = policy_spec.to_dict()

    if isinstance(evaluator_identity, VerifiedEvaluator):
        validate_evaluator_identity(
            evaluator_identity,
            expected_policy_id=str(expected_identity[0][1]),
            expected_policy_source_sha256=str(expected_identity[1][1]),
            expected_policy_abi=(
                str(expected_identity[2][1]) if policy_spec is not None else None
            ),
            expected_policy_parameter_count=int(expected_identity[-1][1]),
        )
    else:
        for key, expected in expected_identity:
            if str(evaluator_identity.get(key, "")).lower() != str(expected).lower():
                raise ValueError(f"evaluator {key} does not match selected grid policy")

    points = expand_grid(grid_spec, policy_spec=policy_spec)
    if policy_spec is None:
        for point in points:
            if point.policy_params:
                raise ValueError(
                    f"native grid point {point.candidate_id} contains policy parameters"
                )
    else:
        policy_profile = profile_from_policy_spec(policy_spec)
        expected_params = len(policy_spec.parameters)
        for point in points:
            if len(point.policy_params) != expected_params:
                raise ValueError(
                    f"grid point {point.candidate_id} has {len(point.policy_params)} policy parameters; "
                    f"{policy_spec.id!r} requires {expected_params}"
                )
            if list(point.policy_params) != quantized(policy_profile, point.policy_params):
                raise ValueError(
                    f"grid point {point.candidate_id} is not on the exact PolicySpec lattice"
                )
            if point.pool_overrides:
                raise ValueError(
                    f"grid point {point.candidate_id} contains pool overrides; "
                    "compiled-policy grids send only dense policy_params"
                )

    run_dir = store.allocate_run_dir("grid", run_id)
    pools = [
        {
            "id": point.candidate_id,
            "ordinal": point.ordinal,
            "coordinate_indices": point.coordinate_indices,
            "coordinates": point.coordinates,
            "coordinate_signature": point.coordinate_signature,
            "policy_params": point.policy_params,
            "pool_overrides": point.pool_overrides,
        }
        for point in points
    ]
    resolved_spec = {
        "grid": grid_spec.to_dict(),
        "pair": pair_spec.to_dict(),
        "scenario": scenario_spec.to_dict(),
        "policy": policy_dict,
        "metric_projection": metric_projection.to_dict(),
    }
    manifest = new_grid_manifest(
        run_id=run_id,
        grid_id=grid_spec.id,
        pool_count=len(points),
        resolved_spec=resolved_spec,
        resolved_axes=[axis.to_dict() for axis in grid_spec.axes],
        pools=pools,
        shards=(),
        core=_core_identity(evaluator_identity),
        artifacts=(),
        table_ref=None,
    )
    manifest_path = run_dir / "manifest.json"
    write_manifest_atomic(manifest_path, manifest, expected_kind="grid")
    return GridCompilation(
        run_dir=run_dir,
        manifest_path=manifest_path,
        points=points,
        manifest=manifest,
    )


def _load_points(manifest: Mapping[str, Any]) -> tuple[GridPoint, ...]:
    grid = manifest.get("grid")
    if not isinstance(grid, Mapping):
        raise GridCoverageError("grid manifest has no grid section")
    raw_points = grid.get("pools")
    if not isinstance(raw_points, Sequence) or isinstance(raw_points, (str, bytes)):
        raise GridCoverageError("grid manifest has no compiled candidate records")
    points: list[GridPoint] = []
    for raw in raw_points:
        if not isinstance(raw, Mapping):
            raise GridCoverageError("compiled grid candidate is not an object")
        policy_params = raw.get("policy_params", ())
        if not isinstance(policy_params, Sequence) or isinstance(policy_params, (str, bytes)):
            raise GridCoverageError("compiled policy_params is not an array")
        coordinates = dict(raw.get("coordinates", {}))
        signature = raw.get("coordinate_signature", "")
        if not isinstance(signature, str):
            raise GridCoverageError("compiled coordinate_signature must be a string")
        if not signature:
            # Manifests written before the signature field remain readable; new
            # compilations always carry the precomputed value.
            signature = coordinate_signature(coordinates)
        points.append(
            GridPoint(
                ordinal=int(raw.get("ordinal", -1)),
                candidate_id=str(raw.get("id", "")),
                coordinate_indices=tuple(int(value) for value in raw.get("coordinate_indices", ())),
                coordinates=coordinates,
                policy_params=tuple(policy_params),
                pool_overrides=dict(raw.get("pool_overrides", {})),
                coordinate_signature=signature,
            )
        )
    points.sort(key=lambda point: point.ordinal)
    if tuple(point.ordinal for point in points) != tuple(range(len(points))):
        raise GridCoverageError("compiled grid ordinals are not a complete canonical range")
    if any(not point.candidate_id for point in points) or len({point.candidate_id for point in points}) != len(points):
        raise GridCoverageError("compiled grid candidate ids are empty or duplicated")
    return tuple(points)


def _metric_projection(manifest: Mapping[str, Any]) -> MetricProjection:
    resolved = manifest.get("resolved_spec", {})
    raw = resolved.get("metric_projection", {}) if isinstance(resolved, Mapping) else {}
    if not isinstance(raw, Mapping) or not raw.get("fields"):
        raise GridCoverageError("grid manifest has no MetricProjection")
    return MetricProjection(
        fields=tuple(str(field) for field in raw["fields"]),
        projection_id=str(raw.get("projection_id", "grid")),
        projection_sha256=str(raw.get("projection_sha256", "")),
    )


def _load_result_records(path: Path) -> tuple[CandidateResult, ...]:
    _, raw_rows = load_grid_results_npz(path)
    results: list[CandidateResult] = []
    for raw in raw_rows:
        transport = dict(raw)
        pool_index = transport.pop("pool_index", None)
        if pool_index is not None and int(pool_index) != int(transport.get("ordinal", -1)):
            raise GridCoverageError(
                f"optimizer pool_index {pool_index!r} does not match evaluator ordinal "
                f"{transport.get('ordinal')!r}"
            )
        results.append(CandidateResult.model_validate(transport))
    return tuple(results)


def collect_grid_run(manifest_path: Path | str, *, results_path: Path | str | None = None, output_path: Path | str | None = None) -> GridRunResult:
    """Collect evaluator results into the common NPZ EvaluationTable."""
    manifest_file = Path(manifest_path).resolve()
    manifest = load_manifest(manifest_file, expected_kind="grid")
    run_dir = manifest_file.parent
    raw_results_path = Path(results_path).resolve() if results_path else run_dir / "grid_results.npz"
    if not raw_results_path.is_file():
        raise FileNotFoundError(f"grid result artifact not found: {raw_results_path}")
    try:
        raw_results_path.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise GridCoverageError("grid result artifact must stay inside the run directory") from exc
    points = _load_points(manifest)
    projection = _metric_projection(manifest)
    resolved = manifest["resolved_spec"]
    grid_spec = resolved.get("grid", {})
    table = collect_evaluations(points, _load_result_records(raw_results_path), metric_projection=projection, metadata={"run_id": manifest["run_id"], "grid_id": manifest["grid"]["grid_id"], "pair_id": resolved.get("pair", {}).get("id"), "scenario_id": resolved.get("scenario", {}).get("id"), "policy_id": resolved.get("policy", {}).get("id"), "shape": list(grid_spec.get("coordinate_shape", ())), "axes": list(grid_spec.get("axes", manifest["grid"].get("resolved_axes", ())))})
    table_path = Path(output_path).resolve() if output_path else run_dir / "evaluation_table.npz"
    try:
        table_path.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise GridCoverageError("evaluation table must stay inside the run directory") from exc
    table.to_npz(table_path)
    artifact_by_path = {str(item.get("path")): dict(item) for item in manifest.get("artifacts", ()) if isinstance(item, Mapping) and item.get("path")}
    for path, kind in ((raw_results_path, "grid_results"), (table_path, "evaluation_table")):
        artifact = _artifact(path, run_dir, kind)
        artifact_by_path[artifact["path"]] = artifact
    manifest["artifacts"] = [artifact_by_path[key] for key in sorted(artifact_by_path)]
    manifest["grid"]["table_ref"] = {
        "path": table_path.relative_to(run_dir).as_posix(),
        "sha256": sha256_path(table_path),
        "bytes": table_path.stat().st_size,
        "row_count": len(table),
        "metric_projection": projection.to_dict(),
    }
    write_manifest_atomic(manifest_file, manifest, expected_kind="grid")
    return GridRunResult(run_dir=run_dir, points=points, table=table, manifest=manifest)


__all__ = ["GridCompilation", "GridRunResult", "collect_grid_run", "compile_grid_run"]
