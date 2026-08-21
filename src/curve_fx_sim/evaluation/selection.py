"""Selection references and strict ReplayPlan normalization."""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, Mapping

from ..artifacts.attestation import (
    load_attested_evaluation_table as load_verified_evaluation_table,
)
from ..grids.model import coordinate_signature
from ..artifacts.store import RunStore
from ..artifacts.tables import EvaluationRow, EvaluationTable
from ..specs.common import (
    SpecError,
    serializable,
)
from ..specs.pair import PairSpec, load_pair_spec
from ..specs.policy import PolicySpec
from ..specs.scenario import ScenarioSpec, load_scenario_spec
from .grouping import SessionGroupKey
from .plans import CandidateCompiler, CandidatePlan
from .session import LocalSessionMaterialization

SelectionKind = Literal["grid_point", "optimizer_winner", "candidate_id"]


@dataclass(frozen=True)
class SelectionRef:
    """A reference to one attested candidate in a grid or optimization run."""

    run_id: str
    kind: SelectionKind = "grid_point"
    index: int | None = None
    coordinate: dict[str, Any] | None = None
    candidate_id: str | None = None
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.run_id:
            raise SpecError("selection run_id must be non-empty")
        if self.kind not in {"grid_point", "optimizer_winner", "candidate_id"}:
            raise SpecError(f"unsupported selection kind: {self.kind!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "kind": self.kind,
            "index": self.index,
            "coordinate": serializable(self.coordinate),
            "candidate_id": self.candidate_id,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SelectionRef:
        unknown = sorted(
            set(data)
            - {"run_id", "kind", "index", "coordinate", "candidate_id", "tags"}
        )
        if unknown:
            raise SpecError("unsupported selection fields: " + ", ".join(unknown))
        return cls(
            run_id=str(data.get("run_id", "")),
            kind=data.get("kind", "grid_point"),  # type: ignore[arg-type]
            index=data.get("index"),
            coordinate=dict(data["coordinate"]) if data.get("coordinate") is not None else None,
            candidate_id=data.get("candidate_id"),
            tags=tuple(data.get("tags", ())),
        )


@dataclass(frozen=True)
class ReplayPlan:
    """A fully resolved, canonical plan for detailed replay, shiftclick diagnostics, or trajectory plotting."""

    run_id: str
    selection: SelectionRef
    pair_spec: PairSpec
    scenario_spec: ScenarioSpec
    policy_id: str
    policy_params: dict[str, Any]
    pool_overrides: dict[str, Any]
    source_row: EvaluationRow
    observation_level: str = "full_trace"
    trace_interval: int = 1

    trace_actions: bool = True
    artifact_dir: Path | None = None
    economic_fingerprint: str | None = None
    compiled_candidate: CandidatePlan | None = None
    tags: tuple[str, ...] = ()
    def __post_init__(self) -> None:
        if self.observation_level not in {"summary", "full_trace"}:
            raise SpecError(f"unsupported replay observation level: {self.observation_level!r}")
        if self.trace_interval <= 0:
            raise SpecError("replay trace_interval must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "selection": self.selection.to_dict(),
            "pair_spec": self.pair_spec.to_dict(),
            "scenario_spec": self.scenario_spec.to_dict(),
            "policy_id": self.policy_id,
            "policy_params": serializable(self.policy_params),
            "pool_overrides": serializable(self.pool_overrides),
            "source_row": self.source_row.to_dict(),
            "observation_level": self.observation_level,
            "trace_interval": self.trace_interval,
            "trace_actions": self.trace_actions,
            "artifact_dir": self.artifact_dir.as_posix() if self.artifact_dir else None,
            "economic_fingerprint": self.economic_fingerprint,
            "compiled_candidate_sha256": self.compiled_candidate.candidate_sha256 if self.compiled_candidate else None,
            "compiled_session_key": self.compiled_candidate.session_key.sha256 if self.compiled_candidate else None,
            "tags": list(self.tags),
        }


def _matches_coordinate(row_coord: Mapping[str, Any] | None, target_coord: Mapping[str, Any]) -> bool:
    """Match one exact decimal coordinate without tolerance or ordinal fallback."""
    return (
        row_coord is not None
        and coordinate_signature(row_coord) == coordinate_signature(target_coord)
    )


def _select_exact_row(table: EvaluationTable, selection: SelectionRef) -> EvaluationRow:
    """Select exactly one row, checking every selector supplied by the caller."""
    if selection.index is None and selection.coordinate is None and selection.candidate_id is None:
        raise SpecError("selection requires an index, coordinate, or candidate_id")

    matches: list[EvaluationRow] = []
    for row in table.rows:
        if selection.index is not None and row.ordinal != selection.index:
            continue
        if selection.coordinate is not None and not _matches_coordinate(row.coordinates, selection.coordinate):
            continue
        if selection.candidate_id is not None and row.candidate_id != selection.candidate_id:
            continue
        matches.append(row)

    contract = "grid point" if selection.kind == "grid_point" else "candidate"
    if not matches:
        if selection.kind == "grid_point" and selection.index is not None:
            raise KeyError(f"grid point ordinal {selection.index} not found")
        if selection.kind == "grid_point" and selection.coordinate is not None:
            raise KeyError(f"grid point coordinate {selection.coordinate!r} not found")
        if selection.candidate_id is not None:
            raise KeyError(f"{contract} candidate_id {selection.candidate_id!r} not found")
        raise KeyError(
            f"{contract} does not match index={selection.index!r}, "
            f"coordinate={selection.coordinate!r}"
        )
    if len(matches) != 1:
        raise KeyError(
            f"{contract} selection is ambiguous; index={selection.index!r}, "
            f"coordinate={selection.coordinate!r}, candidate_id={selection.candidate_id!r}"
        )
    return matches[0]


def _copy_row_values(
    row: EvaluationRow,
    extracted_params: dict[str, Any],
    extracted_overrides: dict[str, Any],
) -> str | None:
    extracted_params.update(row.params)
    extracted_overrides.update(row.pool_overrides)
    return row.economic_fingerprint


def load_attested_evaluation_table(
    manifest: Mapping[str, Any],
    *,
    store: RunStore,
    run_id: str,
) -> EvaluationTable:
    """Load only the evaluation table attested by the run manifest."""
    table, _ = load_verified_evaluation_table(
        manifest,
        run_dir=store.get_run_dir(run_id),
    )
    return table


def _verify_grid_row(manifest: Mapping[str, Any], row: EvaluationRow) -> None:
    """Bind a selected table row to the canonical compiled grid candidate."""
    grid = manifest.get("grid")
    pools = grid.get("pools") if isinstance(grid, Mapping) else None
    if not isinstance(pools, list):
        raise SpecError("grid manifest has no canonical pool candidates")

    matches = [
        pool
        for pool in pools
        if isinstance(pool, Mapping) and pool.get("id") == row.candidate_id
    ]
    if len(matches) != 1:
        raise SpecError(
            f"grid row {row.candidate_id!r} does not identify exactly one canonical pool"
        )
    pool = matches[0]
    if "proposal_evidence" in pool:
        import json

        try:
            candidate = json.loads(str(pool["candidate_json"]))
            expected_params = {"vector": candidate["policy_params"]}
            expected_overrides = candidate["pool_overrides"]
        except (KeyError, TypeError, ValueError) as exc:
            raise SpecError("compiled grid candidate evidence is invalid") from exc
    else:
        expected_params = {"vector": list(pool.get("policy_params", ()))}
        expected_overrides = pool.get("pool_overrides", {})
    checks = (
        ("ordinal", row.ordinal, pool.get("ordinal")),
        ("coordinates", row.coordinates, pool.get("coordinates")),
        ("policy vector", row.params, expected_params),
        ("pool overrides", row.pool_overrides, expected_overrides),
    )
    for label, actual, expected in checks:
        if actual != expected:
            raise SpecError(
                f"grid row {row.candidate_id!r} {label} does not match canonical manifest pool"
            )


def normalize_selection(
    selection: SelectionRef,
    *,
    store: RunStore,
    pair_spec: PairSpec | None = None,
    scenario_spec: ScenarioSpec | None = None,
    policy_spec: PolicySpec | None = None,
    observation_level: str = "full_trace",
    trace_interval: int = 1,
    trace_actions: bool = True,
    artifact_dir: Path | None = None,
    repository: Path | None = None,
    evaluation_table: EvaluationTable | None = None,
) -> ReplayPlan:
    """Normalize a selection using one attested source table and row."""
    root = repository.resolve() if repository is not None else store.root_dir
    manifest = store.load_manifest(selection.run_id)
    resolved_spec = manifest.get("resolved_spec", {})

    # 1. Resolve pair specification, preserving every attested field.
    resolved_pair = pair_spec
    if resolved_pair is None:
        pair_data = resolved_spec.get("pair")
        if isinstance(pair_data, Mapping):
            resolved_pair = PairSpec.from_dict(pair_data)
        elif pair_data:
            resolved_pair = load_pair_spec(str(pair_data), repository=root)
        else:
            raise SpecError(f"cannot resolve pair for run {selection.run_id}")

    # 2. Resolve scenario specification, preserving market files/template metadata.
    resolved_scenario = scenario_spec
    if resolved_scenario is None:
        scen_data = resolved_spec.get("scenario")
        if isinstance(scen_data, Mapping):
            resolved_scenario = ScenarioSpec.from_dict(scen_data)
        elif scen_data:
            resolved_scenario = load_scenario_spec(str(scen_data), repository=root)
        else:
            raise SpecError(
                f"cannot resolve scenario for run {selection.run_id}: "
                "no scenario in resolved_spec and no scenario_spec provided"
            )

    if resolved_scenario.pair_id != resolved_pair.id:
        raise SpecError(
            f"scenario {resolved_scenario.id} pair {resolved_scenario.pair_id!r} "
            f"does not match pair {resolved_pair.id!r}"
        )

    # 3. Resolve policy ID
    policy_id = ""
    if policy_spec is not None:
        policy_id = policy_spec.id
    elif "policy" in resolved_spec and isinstance(resolved_spec["policy"], Mapping):
        policy_id = resolved_spec["policy"].get("id", "")
    elif "policy" in resolved_spec and isinstance(resolved_spec["policy"], str):
        policy_id = resolved_spec["policy"]
    elif "policy_id" in manifest.get("core", {}):
        policy_id = manifest["core"]["policy_id"] or ""
    elif "grid" in manifest and "policy_id" in manifest["grid"]:
        policy_id = manifest["grid"]["policy_id"] or ""
    elif "optimization" in manifest and "policy_id" in manifest["optimization"]:
        policy_id = manifest["optimization"]["policy_id"] or ""

    # 4. Extract parameters and overrides based on selection kind.
    # Callers that already attested the table pass it in so replay paths do not
    # parse the same large artifact again.
    table = evaluation_table
    extracted_params: dict[str, Any] = {}
    extracted_overrides: dict[str, Any] = {}
    economic_fingerprint: str | None = None
    source_row: EvaluationRow | None = None
    if selection.kind == "grid_point":
        if (
            selection.index is None
            and selection.coordinate is None
            and selection.candidate_id is None
        ):
            raise SpecError("grid_point selection requires an index, coordinate, or candidate_id")

        if table is None:
            table = load_attested_evaluation_table(
                manifest, store=store, run_id=selection.run_id
            )
        found_row = _select_exact_row(table, selection)
        source_row = found_row
        if manifest.get("run_kind") != "grid":
            raise SpecError("grid_point selection requires a grid run")
        _verify_grid_row(manifest, found_row)
        economic_fingerprint = _copy_row_values(
            found_row, extracted_params, extracted_overrides
        )

    elif selection.kind == "candidate_id":
        if not selection.candidate_id:
            raise SpecError("candidate_id selection requires 'candidate_id'")

        if table is None:
            table = load_attested_evaluation_table(
                manifest, store=store, run_id=selection.run_id
            )
        found_row = _select_exact_row(table, selection)
        source_row = found_row
        if manifest.get("run_kind") == "grid":
            _verify_grid_row(manifest, found_row)
        economic_fingerprint = _copy_row_values(
            found_row, extracted_params, extracted_overrides
        )

    elif selection.kind == "optimizer_winner":
        opt_info = manifest.get("optimization", {})
        best_cand = opt_info.get("best_candidate")
        if not isinstance(best_cand, Mapping):
            raise SpecError(
                f"run {selection.run_id} has no immutable optimization best_candidate"
            )

        best_id = best_cand.get("candidate_id")
        if not isinstance(best_id, str) or not best_id:
            raise SpecError(
                f"run {selection.run_id} has no persisted optimizer winner candidate_id"
            )
        if selection.candidate_id is not None and best_id != selection.candidate_id:
            raise KeyError(
                f"requested candidate_id {selection.candidate_id!r} is not the persisted "
                f"optimizer winner {best_id!r}"
            )

        best_params = best_cand.get("params", {})
        best_overrides = best_cand.get("pool_overrides", {})
        if not isinstance(best_params, Mapping) or not isinstance(best_overrides, Mapping):
            raise SpecError(
                f"run {selection.run_id} has invalid optimizer winner parameters"
            )
        extracted_params = dict(best_params)
        extracted_overrides = dict(best_overrides)
        economic_fingerprint = best_cand.get("economic_fingerprint")

        expected_lineage = best_cand.get("lineage")
        if isinstance(expected_lineage, Mapping) and selection.coordinate is not None:
            if dict(selection.coordinate) != dict(expected_lineage):
                raise KeyError("requested optimizer winner coordinate does not match manifest")

        if table is None:
            table = load_attested_evaluation_table(
                manifest, store=store, run_id=selection.run_id
            )
        row_selector = SelectionRef(
            run_id=selection.run_id,
            kind="optimizer_winner",
            index=selection.index,
            coordinate=selection.coordinate,
            candidate_id=best_id,
        )
        found_row = _select_exact_row(table, row_selector)
        source_row = found_row
        if expected_lineage is not None and (
            found_row.coordinates is None
            or found_row.coordinates != expected_lineage
        ):
            raise SpecError(
                f"optimizer winner row {best_id!r} lineage does not match manifest"
            )
        if found_row.params != dict(best_params) or found_row.pool_overrides != dict(
            best_overrides
        ):
            raise SpecError(
                f"optimizer winner row {best_id!r} parameters do not match manifest"
            )
        row_fingerprint = found_row.economic_fingerprint
        if economic_fingerprint is not None and row_fingerprint != economic_fingerprint:
            raise SpecError(
                f"optimizer winner row {best_id!r} economic fingerprint does not match manifest"
            )
        economic_fingerprint = row_fingerprint
    else:
        raise SpecError(f"unsupported selection kind: {selection.kind!r}")

    if source_row is None:
        raise SpecError("selection did not resolve an attested evaluation row")

    target_artifact_dir = artifact_dir
    if target_artifact_dir is None:
        target_artifact_dir = store.get_run_dir(selection.run_id) / "replay"

    return ReplayPlan(
        run_id=selection.run_id,
        selection=selection,
        pair_spec=resolved_pair,
        scenario_spec=resolved_scenario,
        policy_id=policy_id,
        policy_params=extracted_params,
        pool_overrides=extracted_overrides,
        source_row=source_row,
        observation_level=observation_level,
        trace_interval=trace_interval,
        trace_actions=trace_actions,
        artifact_dir=target_artifact_dir,
        economic_fingerprint=economic_fingerprint,
        tags=selection.tags,
    )


def compile_selected_replay(
    plan: ReplayPlan,
    *,
    manifest: Mapping[str, Any],
    selected_evaluator: Any,
    materialization: LocalSessionMaterialization,
    yb_off: bool = False,
) -> ReplayPlan:
    """Recompile one selected row through its run-local evaluator evidence."""
    compiler = selected_evaluator.compiler
    row = plan.source_row
    if manifest.get("run_kind") == "grid":
        from ..grids.runner import load_grouped_grid

        points, _ = load_grouped_grid(
            manifest, parameter_schema=compiler.schema,
            artifact_sha256=selected_evaluator.artifact_sha256)
        matches = [point for point in points if point.ordinal == row.ordinal and point.candidate_id == row.candidate_id]
        if len(matches) != 1 or matches[0].evaluation is None:
            raise SpecError("selected grid row has no exact compiled candidate evidence")
        point = matches[0]
        proposal = point.proposal_dict
        expected_candidate_sha = point.evaluation.candidate.candidate_sha256
        expected_group = point.session_group_id
        if point.evaluation.evaluation_id != row.candidate_id:
            raise SpecError("selected grid evaluation ID does not match its table row")
    elif manifest.get("run_kind") == "optimization":
        named = row.params.get("named")
        lineage = row.params.get("evaluation_lineage")
        if not isinstance(named, Mapping) or not isinstance(lineage, list):
            raise SpecError("selected optimizer row has no named proposal or evaluation lineage")
        matches = [item for item in lineage if isinstance(item, Mapping) and item.get("evaluation_id") == row.candidate_id]
        if len(matches) != 1:
            raise SpecError("selected optimizer row has no exact primary evaluation lineage")
        evidence = matches[0]
        if evidence.get("scenario_id") != plan.scenario_spec.id:
            raise SpecError("selected optimizer evaluation lineage has the wrong scenario")
        proposal = dict(named)
        expected_candidate_sha = str(evidence.get("candidate_sha256", ""))
        expected_group = str(evidence.get("session_group_id", ""))
        if row.params.get("candidate_sha256") != expected_candidate_sha:
            raise SpecError("selected optimizer candidate SHA differs from its lineage")
    else:
        raise SpecError("artifact replay requires a grid or optimization source run")

    compiled_proposal: dict[str, object] = {}
    for name, value in proposal.items():
        descriptor = compiler.schema.descriptor(name)
        if descriptor.classification == "observation" and descriptor.lowering_path.startswith("evaluate_batch."):
            continue
        value_type = descriptor.value_type
        if value_type == "real" and isinstance(value, str):
            value = Decimal(value)
        elif value_type == "real_pair" and isinstance(value, (list, tuple)):
            value = tuple(Decimal(item) if isinstance(item, str) else item for item in value)
        compiled_proposal[name] = value
    if yb_off:
        compiler.schema.descriptor("run.yb_mode")
        compiled_proposal["run.yb_mode"] = "off"
        compiled_proposal.pop("run.yb_releverage", None)
    candidate = compiler.compile(
        compiled_proposal, open_session=materialization.baseline_open_session_fields,
        scenario=materialization.closure)
    if candidate.candidate_sha256 != expected_candidate_sha:
        raise SpecError("selected proposal recompiles to a different candidate SHA")
    group = SessionGroupKey.create(
        selected_evaluator.artifact_sha256, compiler.schema,
        candidate.scenario_key, candidate.session_key).validated()
    if not yb_off and group.sha256 != expected_group:
        raise SpecError("selected proposal recompiles to a different SessionGroup")
    return replace(
        plan, policy_params={"vector": list(candidate.policy_params)},
        pool_overrides=candidate.pool_overrides, compiled_candidate=candidate)


__all__ = [
    "SelectionKind",
    "SelectionRef",
    "ReplayPlan",
    "load_attested_evaluation_table",
    "compile_selected_replay",
    "normalize_selection",
]
