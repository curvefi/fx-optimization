"""Finite-grid compilation and shared artifact collection."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from curve_fx_harness_client.models import CandidateResult

from ..artifacts.io import sha256_path
from ..execution.collection import load_grid_results_npz
from ..artifacts.manifest import load_manifest, new_grid_manifest, write_manifest_atomic
from ..artifacts.store import RunStore
from ..artifacts.tables import EvaluationTable, MetricProjection
from ..evaluation.grouping import (
    CompiledEvaluation,
    PortableCandidate,
    SessionGroup,
    SessionGroupKey,
    group_evaluations,
)
from ..evaluation.plans import CandidateSchema, ObservationKey, ScenarioKey, SessionKey
from ..evaluation.selected import SelectedEvaluator, materialize_selected_evaluator
from ..evaluation.identity import VerifiedEvaluator, validate_evaluator_identity
from ..specs.common import canonical_json_bytes
from ..specs.grid import GridSpec
from ..specs.pair import PairSpec
from ..specs.policy import PolicySpec
from ..specs.scenario import ScenarioClosure, ScenarioSpec
from ..optimization.profiles import profile_from_policy_spec, quantized
from .collection import GridCoverageError, collect_evaluations
from .model import GridPoint, compile_grid_points, coordinate_signature, expand_grid

_GROUPED_GRID_MODE = "schema_grouped_v1"


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


def _key_record(key: Any) -> dict[str, str]:
    return {"sha256": key.sha256, "identity_json": key.identity_json.decode("utf-8")}


def encode_session_groups(groups: Sequence[SessionGroup]) -> list[dict[str, Any]]:
    return [
        {
            "session_group_id": group.key.sha256,
            "session_group_key": _key_record(group.key),
            "scenario_key": _key_record(group.scenario_key),
            "session_key": _key_record(group.session_key),
            "observations": [
                {
                    "observation_id": observation.key.sha256,
                    "observation_key": _key_record(observation.key),
                    "ordinals": [item.ordinal for item in observation.evaluations],
                }
                for observation in group.observation_groups
            ],
        }
        for group in groups
    ]


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
    evaluator_identity: VerifiedEvaluator | Mapping[str, Any] | None = None,
    selected_evaluator: SelectedEvaluator | None = None,
    open_session: Mapping[str, object] | None = None,
    scenario: ScenarioClosure | ScenarioKey | None = None,
) -> GridCompilation:
    """Compile one finite grid, preserving the explicit legacy call path.

    Supplying one ``SelectedEvaluator``, ``open_session``, and ``scenario``
    as a complete set selects schema-backed named proposals. Omitting all
    three retains raw ``AxisTarget`` lowering for legacy workflows only.
    """
    _validate_specs(grid_spec, pair_spec, scenario_spec, policy_spec, metric_projection)

    schema_inputs = (selected_evaluator, open_session, scenario)
    if any(value is not None for value in schema_inputs) and not all(
        value is not None for value in schema_inputs
    ):
        raise ValueError(
            "schema-backed grid compilation requires selected_evaluator, open_session, "
            "and scenario together"
        )
    schema_mode = selected_evaluator is not None
    if schema_mode and evaluator_identity is not None:
        raise ValueError("schema-backed grid compilation cannot mix evaluator_identity")
    if not schema_mode and evaluator_identity is None:
        raise ValueError("legacy grid compilation requires evaluator_identity")
    if selected_evaluator is not None and not isinstance(
        selected_evaluator, SelectedEvaluator
    ):
        raise TypeError("selected_evaluator must be a SelectedEvaluator")
    core = (
        selected_evaluator.manifest_core()
        if selected_evaluator is not None
        else _core_identity(evaluator_identity)
    )

    if schema_mode:
        assert selected_evaluator is not None
        compiler = selected_evaluator.compiler
        if selected_evaluator.parameter_schema_sha256 != compiler.schema.sha256:
            raise ValueError("selected evaluator compiler schema does not match its artifact")
        provenance = selected_evaluator.provenance
        if (
            provenance.get("binary_sha256") != core.get("sha256")
            or provenance.get("parameter_schema_sha256") != compiler.schema.sha256
        ):
            raise ValueError("selected evaluator provenance does not match its manifest core")
        if provenance.get("policy") != {
            "id": core.get("policy_id"),
            "abi": core.get("policy_abi"),
            "source_sha256": core.get("policy_source_sha256"),
        }:
            raise ValueError("selected evaluator policy provenance does not match its core")
        policy_descriptors = tuple(
            item for item in compiler.schema.descriptors if item.order is not None
        )
        expected_count = len(policy_descriptors)
        expected_policy_id = policy_spec.id if policy_spec is not None else compiler.schema.policy_id
        if policy_spec is None and expected_count:
            raise ValueError(
                "passthrough grid evaluator description declares policy parameters"
            )
        if compiler.schema.policy_id != expected_policy_id:
            raise ValueError(
                f"evaluator description policy {compiler.schema.policy_id!r} does not match "
                f"selected grid policy {expected_policy_id!r}"
            )
        expected_identity = [
            ("policy_id", expected_policy_id),
            ("policy_parameter_count", expected_count),
        ]
        if policy_spec is not None:
            expected_identity.extend(
                (
                    ("policy_source_sha256", policy_spec.source_sha256),
                    ("policy_abi", policy_spec.policy_abi),
                )
            )
            policy_dict = policy_spec.to_dict()
        else:
            policy_dict = {
                "id": compiler.schema.policy_id,
                "policy_kind": "compiled_passthrough",
                "source_sha256": core.get("policy_source_sha256"),
                "policy_abi": core.get("policy_abi"),
                "parameters": [],
            }
    elif policy_spec is None:
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

    identity_authority = core if schema_mode else evaluator_identity
    if isinstance(identity_authority, VerifiedEvaluator):
        expected = dict(expected_identity)
        validate_evaluator_identity(
            identity_authority,
            expected_policy_id=str(expected["policy_id"]),
            expected_policy_source_sha256=(
                str(expected["policy_source_sha256"])
                if "policy_source_sha256" in expected
                else None
            ),
            expected_policy_abi=(str(expected["policy_abi"]) if "policy_abi" in expected else None),
            expected_policy_parameter_count=int(expected["policy_parameter_count"]),
        )
    else:
        assert isinstance(identity_authority, Mapping)
        for key, expected in expected_identity:
            if str(identity_authority.get(key, "")).lower() != str(expected).lower():
                raise ValueError(f"evaluator {key} does not match selected grid policy")

    if schema_mode:
        assert (
            selected_evaluator is not None
            and open_session is not None
            and scenario is not None
        )
        compiler = selected_evaluator.compiler
        points = compile_grid_points(
            grid_spec,
            compiler=compiler,
            artifact_sha256=selected_evaluator.artifact_sha256,
            open_session=open_session,
            scenario=scenario,
        )
        groups = group_evaluations(
            tuple(point.evaluation for point in points if point.evaluation is not None),
            artifact_sha256=selected_evaluator.artifact_sha256,
            parameter_schema=compiler.schema,
        )
    else:
        points = expand_grid(grid_spec, policy_spec=policy_spec)
    if not schema_mode and policy_spec is None:
        for point in points:
            if point.policy_params:
                raise ValueError(
                    f"native grid point {point.candidate_id} contains policy parameters"
                )
    elif not schema_mode:
        assert policy_spec is not None
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
    artifact_records: tuple[dict[str, Any], ...] = ()
    if schema_mode:
        assert selected_evaluator is not None
        selected_evaluator = materialize_selected_evaluator(
            selected_evaluator, run_dir, resume=False
        )
        if selected_evaluator.provenance != provenance:
            raise ValueError(
                "run-local evaluator selection differs from compiled grid selection"
            )
        core = selected_evaluator.manifest_core(
            binary_override="evaluator_artifact/evaluator"
        )
        artifact_records = (
            _artifact(
                run_dir / "evaluator_artifact" / "artifact.json",
                run_dir,
                "evaluator_artifact_receipt",
            ),
            _artifact(
                run_dir / "evaluator_artifact" / "evaluator",
                run_dir,
                "evaluator_binary",
            ),
        )
    pools = []
    for point in points:
        record = point.to_dict()
        record["id"] = record.pop("candidate_id")
        if schema_mode:
            if not record.get("evaluation_id"):
                raise ValueError("schema-backed grid point has no evaluation reference")
        pools.append(record)
    resolved_spec = {
        "grid": grid_spec.to_dict(),
        "pair": pair_spec.to_dict(),
        "scenario": scenario_spec.to_dict(),
        "policy": policy_dict,
        "metric_projection": metric_projection.to_dict(),
    }
    if schema_mode:
        assert selected_evaluator is not None
        compiler = selected_evaluator.compiler
        resolved_spec["evaluator_artifact_selection"] = selected_evaluator.provenance
        resolved_spec["candidate_compilation"] = {
            "mode": _GROUPED_GRID_MODE,
            "parameter_schema_version": compiler.schema.schema_version,
            "parameter_schema_sha256": compiler.schema.sha256,
            "policy_id": compiler.schema.policy_id,
            "groups": encode_session_groups(groups),
        }
    manifest = new_grid_manifest(
        run_id=run_id,
        grid_id=grid_spec.id,
        pool_count=len(points),
        resolved_spec=resolved_spec,
        resolved_axes=[axis.to_dict() for axis in grid_spec.axes],
        pools=pools,
        shards=(),
        core=core,
        artifacts=artifact_records,
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


def load_grouped_grid(
    manifest: Mapping[str, Any],
    *,
    parameter_schema: CandidateSchema,
    artifact_sha256: str,
) -> tuple[tuple[GridPoint, ...], tuple[SessionGroup, ...]]:
    grid = manifest.get("grid")
    if not isinstance(grid, Mapping):
        raise GridCoverageError("grid manifest has no grid section")
    raw_points = grid.get("pools")
    if not isinstance(raw_points, Sequence) or isinstance(raw_points, (str, bytes)):
        raise GridCoverageError("grid manifest has no compiled candidate records")
    resolved = manifest.get("resolved_spec", {})
    compilation = (
        resolved.get("candidate_compilation", {})
        if isinstance(resolved, Mapping)
        else {}
    )
    if not isinstance(compilation, Mapping):
        raise GridCoverageError("resolved candidate_compilation must be an object")
    mode = compilation.get("mode")
    if mode != _GROUPED_GRID_MODE:
        raise GridCoverageError(f"unsupported grouped candidate_compilation mode {mode!r}")

    ordinal_groups: dict[int, tuple[str, ScenarioKey, SessionKey, ObservationKey]] = {}
    raw_groups = compilation.get("groups")
    if not isinstance(raw_groups, list) or not raw_groups:
        raise GridCoverageError("schema-grouped manifest has no session groups")
    for raw_group in raw_groups:
            if not isinstance(raw_group, Mapping):
                raise GridCoverageError("compiled session group is not an object")
            group_key = _load_key(raw_group.get("session_group_key"), SessionGroupKey)
            scenario_key = _load_key(raw_group.get("scenario_key"), ScenarioKey)
            session_key = _load_key(raw_group.get("session_key"), SessionKey)
            if not all((group_key, scenario_key, session_key)):
                raise GridCoverageError("compiled session group key evidence is incomplete")
            assert isinstance(group_key, SessionGroupKey)
            assert isinstance(scenario_key, ScenarioKey)
            assert isinstance(session_key, SessionKey)
            expected_key = SessionGroupKey.create(
                artifact_sha256, parameter_schema, scenario_key, session_key
            ).validated()
            if group_key != expected_key or raw_group.get("session_group_id") != group_key.sha256:
                raise GridCoverageError("compiled session group identity is inconsistent")
            observations = raw_group.get("observations")
            if not isinstance(observations, list) or not observations:
                raise GridCoverageError("compiled session group has no observations")
            for raw_observation in observations:
                if not isinstance(raw_observation, Mapping):
                    raise GridCoverageError("compiled observation group is not an object")
                observation_key = _load_key(
                    raw_observation.get("observation_key"), ObservationKey
                )
                if not isinstance(observation_key, ObservationKey):
                    raise GridCoverageError("compiled observation key evidence is incomplete")
                observation_key.validated(parameter_schema)
                if raw_observation.get("observation_id") != observation_key.sha256:
                    raise GridCoverageError("compiled observation identity is inconsistent")
                ordinals = raw_observation.get("ordinals")
                if not isinstance(ordinals, list) or any(
                    isinstance(value, bool) or not isinstance(value, int) or value < 0
                    for value in ordinals
                ):
                    raise GridCoverageError("compiled observation ordinals are invalid")
                for ordinal in ordinals:
                    if ordinal in ordinal_groups:
                        raise GridCoverageError("compiled group ordinals overlap")
                    ordinal_groups[ordinal] = (
                        group_key.sha256, scenario_key, session_key, observation_key
                    )
    points: list[GridPoint] = []
    for raw in raw_points:
        if not isinstance(raw, Mapping):
            raise GridCoverageError("compiled grid candidate is not an object")
        coordinates = dict(raw.get("coordinates", {}))
        signature = raw.get("coordinate_signature", "")
        if not isinstance(signature, str):
            raise GridCoverageError("compiled coordinate_signature must be a string")
        if not signature:
            # Manifests written before the signature field remain readable; new
            # compilations always carry the precomputed value.
            signature = coordinate_signature(coordinates)
        proposal_evidence = raw.get("proposal_evidence")
        if not isinstance(proposal_evidence, Mapping):
            raise GridCoverageError("named proposal evidence must be an object")
        raw_proposal = proposal_evidence
        candidate_json = raw.get("candidate_json", "")
        if not isinstance(candidate_json, str):
            raise GridCoverageError("compiled candidate JSON evidence must be a string")
        candidate_sha256 = str(raw.get("candidate_sha256", ""))
        candidate = _validate_candidate_evidence(candidate_json, candidate_sha256)
        policy_params = candidate["policy_params"]
        raw_pool_overrides = candidate["pool_overrides"]
        ordinal = int(raw.get("ordinal", -1))
        try:
            group_id, scenario_key, session_key, observation_key = ordinal_groups[ordinal]
        except KeyError as exc:
            raise GridCoverageError("compiled pool has no group observation reference") from exc
        evaluation_id = str(raw.get("evaluation_id", ""))
        if (
            evaluation_id != raw.get("id")
            or raw.get("session_group_id") != group_id
            or raw.get("observation_id") != observation_key.sha256
        ):
            raise GridCoverageError("compiled pool references the wrong evaluation/group/observation")
        evaluation = CompiledEvaluation(
                PortableCandidate(
                    scenario_key,
                    session_key,
                    tuple(policy_params),
                    canonical_json_bytes(dict(raw_pool_overrides)),
                    candidate_json.encode("utf-8"),
                    candidate_sha256,
                ),
                artifact_sha256,
                observation_key,
                ordinal,
                evaluation_id,
            )
        points.append(
            GridPoint(
                ordinal=ordinal,
                candidate_id=str(raw.get("id", "")),
                coordinate_indices=tuple(int(value) for value in raw.get("coordinate_indices", ())),
                coordinates=coordinates,
                legacy_policy_params=(),
                legacy_pool_overrides={},
                coordinate_signature=signature,
                # Decimal proposal values persist as exact strings.  Preserve
                # them for audit display only; they are not compiler input.
                proposal=tuple(
                    (str(name), _freeze_manifest_value(value))
                    for name, value in sorted(raw_proposal.items())
                ),
                session_group_id=group_id,
                evaluation=evaluation,
            )
        )
    points.sort(key=lambda point: point.ordinal)
    if tuple(point.ordinal for point in points) != tuple(range(len(points))):
        raise GridCoverageError("compiled grid ordinals are not a complete canonical range")
    if any(not point.candidate_id for point in points) or len({point.candidate_id for point in points}) != len(points):
        raise GridCoverageError("compiled grid candidate ids are empty or duplicated")
    if set(ordinal_groups) != set(range(len(points))):
        raise GridCoverageError("compiled groups do not exactly cover grid ordinals")
    groups = group_evaluations(
        tuple(point.evaluation for point in points),
        artifact_sha256=artifact_sha256,
        parameter_schema=parameter_schema,
    )
    if encode_session_groups(groups) != compilation.get("groups"):
        raise GridCoverageError("compiled groups do not match canonical evaluation grouping")
    return tuple(points), groups


def _validate_candidate_evidence(
    candidate_json: str,
    candidate_sha256: str,
) -> dict[str, Any]:
    if not candidate_json or not candidate_sha256:
        raise GridCoverageError("compiled candidate JSON and SHA-256 must appear together")
    encoded = candidate_json.encode("utf-8")
    if hashlib.sha256(encoded).hexdigest() != candidate_sha256:
        raise GridCoverageError("compiled candidate JSON does not match candidate_sha256")
    try:
        payload = json.loads(
            candidate_json,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise GridCoverageError("compiled candidate JSON is not strict JSON") from exc
    if not isinstance(payload, Mapping) or set(payload) != {"policy_params", "pool_overrides"}:
        raise GridCoverageError("compiled candidate JSON has the wrong payload shape")
    if canonical_json_bytes(payload) != encoded:
        raise GridCoverageError("compiled candidate JSON is not canonical")
    if not isinstance(payload["policy_params"], list) or not isinstance(
        payload["pool_overrides"], Mapping
    ):
        raise GridCoverageError("compiled candidate JSON has invalid candidate fields")
    return dict(payload)


def _freeze_manifest_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(
            (str(key), _freeze_manifest_value(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, list):
        return tuple(_freeze_manifest_value(item) for item in value)
    return value


def _load_key(
    value: Any,
    key_type: type[Any],
) -> Any | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise GridCoverageError("compiled scenario/session key evidence must be an object")
    sha256 = value.get("sha256")
    identity_json = value.get("identity_json")
    if not isinstance(sha256, str) or not isinstance(identity_json, str):
        raise GridCoverageError("compiled scenario/session key evidence is incomplete")
    if hashlib.sha256(identity_json.encode("utf-8")).hexdigest() != sha256:
        raise GridCoverageError("compiled scenario/session identity does not match its SHA-256")
    return key_type(identity_json=identity_json.encode("utf-8"), sha256=sha256)


def _load_legacy_points(manifest: Mapping[str, Any]) -> tuple[GridPoint, ...]:
    grid = manifest.get("grid")
    raw_points = grid.get("pools") if isinstance(grid, Mapping) else None
    if not isinstance(raw_points, list):
        raise GridCoverageError("legacy grid manifest has no pool records")
    points = tuple(
        GridPoint(
            ordinal=int(raw["ordinal"]),
            candidate_id=str(raw["id"]),
            coordinate_indices=tuple(int(value) for value in raw.get("coordinate_indices", ())),
            coordinates=dict(raw.get("coordinates", {})),
            legacy_policy_params=tuple(raw.get("policy_params", ())),
            legacy_pool_overrides=dict(raw.get("pool_overrides", {})),
            coordinate_signature=str(raw.get("coordinate_signature", ""))
            or coordinate_signature(dict(raw.get("coordinates", {}))),
        )
        for raw in raw_points
        if isinstance(raw, Mapping)
    )
    if len(points) != len(raw_points) or tuple(point.ordinal for point in points) != tuple(
        range(len(points))
    ):
        raise GridCoverageError("legacy grid points do not have canonical coverage")
    return points


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
    resolved = manifest.get("resolved_spec")
    compilation = resolved.get("candidate_compilation") if isinstance(resolved, Mapping) else None
    if isinstance(compilation, Mapping) and compilation.get("mode") == _GROUPED_GRID_MODE:
        try:
            selected = SelectedEvaluator.load(run_dir / "evaluator_artifact")
        except Exception as exc:
            raise GridCoverageError(
                f"cannot verify grouped grid evaluator artifact: {exc}"
            ) from exc
        points, _ = load_grouped_grid(
            manifest,
            parameter_schema=selected.compiler.schema,
            artifact_sha256=selected.artifact_sha256,
        )
    else:
        points = _load_legacy_points(manifest)
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


__all__ = [
    "GridCompilation",
    "GridRunResult",
    "collect_grid_run",
    "compile_grid_points",
    "compile_grid_run",
    "encode_session_groups",
    "load_grouped_grid",
]
