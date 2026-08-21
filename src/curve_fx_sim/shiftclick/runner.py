"""Strict SelectionRef -> ReplayPlan shiftclick execution and verification."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from ..analysis.economics import EconomicComparison, compare_economics
from ..artifacts.io import atomic_write_json, sha256_path
from ..artifacts.manifest import new_shiftclick_manifest, write_manifest_atomic
from ..artifacts.store import RunStore
from ..evaluation.selection import (
    ReplayPlan,
    SelectionRef,
    compile_selected_replay,
    load_attested_evaluation_table,
    normalize_selection,
)
from ..evaluation.selected import SelectedEvaluator
from ..evaluation.session import LocalSessionMaterialization
from ..evaluation.grouping import CompiledEvaluation, bind_local_session_group, group_evaluations
from ..execution.grouped import execute_local_groups
from ..specs.scenario import ScenarioSpec
from ..specs.shiftclick import ShiftclickSpec
from ..specs.common import assert_contained_path
from .archive import pack_replay_archive


class ShiftclickError(ValueError):
    """Raised when a shiftclick source or full replay fails closed."""


@dataclass(frozen=True)
class ReplayObservationPolicy:
    """One replay policy for exact source-YB and sparse no-YB inspection."""

    mode: Literal["source_yb", "yb_disabled_sparse"] = "source_yb"

    @classmethod
    def from_spec(cls, spec: ShiftclickSpec) -> "ReplayObservationPolicy":
        requested = {
            tag.removeprefix("observation:")
            for tag in spec.tags
            if tag.startswith("observation:")
        }
        if len(requested) > 1:
            raise ShiftclickError("shiftclick declares conflicting observation policies")
        value = next(iter(requested), "source-yb")
        if value == "source-yb":
            return cls("source_yb")
        if value == "yb-disabled":
            return cls("yb_disabled_sparse")
        raise ShiftclickError(f"unsupported shiftclick observation policy {value!r}")

    @property
    def compare_to_source(self) -> bool:
        return self.mode == "source_yb"

    def scenario(self, source: ScenarioSpec) -> ScenarioSpec:
        if self.mode == "source_yb":
            return source
        return replace(source, yb_mode="off", yb_releverage=False)

    def compare(
        self,
        expected_metrics: Mapping[str, Any],
        observed_metrics: Mapping[str, Any],
        *,
        expected_fingerprint: str,
        observed_fingerprint: str,
        fields: Sequence[str],
    ) -> tuple[EconomicComparison | None, dict[str, Any]]:
        if self.compare_to_source:
            comparison = compare_economics(
                expected_metrics,
                observed_metrics,
                expected_fingerprint=expected_fingerprint,
                observed_fingerprint=observed_fingerprint,
                fields=fields,
            )
            return comparison, comparison.to_dict()
        if not expected_fingerprint or not observed_fingerprint:
            raise ShiftclickError("counterfactual replay requires both economic fingerprints")
        return None, {
            "status": "counterfactual",
            "reason": "YieldBasis is disabled for sparse inspection",
            "source_economic_fingerprint": expected_fingerprint,
            "replay_economic_fingerprint": observed_fingerprint,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "yb_mode": "source" if self.compare_to_source else "off",
            "target_trace_points": None if self.compare_to_source else 10_000,
            "economic_comparison": "exact" if self.compare_to_source else "counterfactual",
        }


@dataclass(frozen=True)
class ShiftclickResult:
    run_dir: Path
    plan: ReplayPlan
    source_candidate_id: str
    replay_result: Mapping[str, Any]
    comparison: EconomicComparison | None
    comparison_receipt: Mapping[str, Any]
    manifest: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_dir": self.run_dir.as_posix(),
            "plan": self.plan.to_dict(),
            "source_candidate_id": self.source_candidate_id,
            "replay_result": dict(self.replay_result),
            "comparison": dict(self.comparison_receipt),
            "manifest": dict(self.manifest),
        }



def selection_from_spec(spec: ShiftclickSpec) -> SelectionRef:
    """Construct the only allowed source reference; never infer by row order."""
    source_kind = spec.source_kind
    kind = spec.selection_kind
    value = spec.selection_value
    if source_kind == "optimization" and kind == "best":
        return SelectionRef(run_id=spec.source_run_id, kind="optimizer_winner", tags=spec.tags)
    if source_kind == "grid" and kind == "best":
        raise ShiftclickError("grid selection 'best' is ambiguous; provide coordinates, index, or candidate_id")
    if kind == "coordinates":
        if not isinstance(value, Mapping) or not value:
            raise ShiftclickError("coordinate selection requires a non-empty mapping")
        return SelectionRef(run_id=spec.source_run_id, kind="grid_point", coordinate=dict(value), tags=spec.tags)
    if kind == "index":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ShiftclickError("index selection requires an integer")
        index = value
        return SelectionRef(run_id=spec.source_run_id, kind="grid_point", index=index, tags=spec.tags)
    if kind == "candidate_id":
        if not isinstance(value, str) or not value:
            raise ShiftclickError("candidate_id selection requires a non-empty string")
        return SelectionRef(
            run_id=spec.source_run_id,
            kind="candidate_id",
            candidate_id=value,
            tags=spec.tags,
        )
    raise ShiftclickError(f"unsupported shiftclick selection_kind {kind!r}")


def _source_scenarios(manifest: Mapping[str, Any], primary: ScenarioSpec) -> tuple[ScenarioSpec, ...]:
    resolved = manifest.get("resolved_spec", {})
    raw = resolved.get("scenario_specs") if isinstance(resolved, Mapping) else None
    scenarios = (primary,) if raw is None else tuple(ScenarioSpec.from_dict(item) for item in raw)
    if not scenarios or len({item.id for item in scenarios}) != len(scenarios) or primary.id != scenarios[0].id:
        raise ShiftclickError("source scenario_specs are empty, duplicated, or reorder the primary scenario")
    return scenarios


def _evaluation_ids(plan: ReplayPlan, scenarios: Sequence[ScenarioSpec], run_kind: str) -> tuple[str, ...]:
    if run_kind == "grid":
        if len(scenarios) != 1:
            raise ShiftclickError("grid replay must contain exactly one scenario")
        return (plan.source_row.candidate_id,)
    lineage = plan.source_row.params.get("evaluation_lineage")
    if not isinstance(lineage, list):
        raise ShiftclickError("optimizer replay has no evaluation lineage")
    by_scenario = {item.get("scenario_id"): item.get("evaluation_id") for item in lineage if isinstance(item, Mapping)}
    result = tuple(by_scenario.get(scenario.id) for scenario in scenarios)
    if any(not isinstance(value, str) or not value for value in result) or len(set(result)) != len(result):
        raise ShiftclickError("optimizer replay lineage lacks exact scenario coverage")
    return result  # type: ignore[return-value]


def _artifact(path: Path, run_dir: Path, kind: str) -> dict[str, Any]:
    return {"path": path.relative_to(run_dir).as_posix(), "kind": kind,
            "bytes": path.stat().st_size, "sha256": sha256_path(path)}


def run_shiftclick(
    spec: ShiftclickSpec,
    *,
    store: RunStore,
    selection: SelectionRef | None = None,
    output_dir: Path | None = None,
) -> ShiftclickResult:
    """Replay one artifact-selected candidate across its ordered source scenarios."""
    if selection is None:
        selection = selection_from_spec(spec)
    else:
        if selection.run_id != spec.source_run_id:
            raise ShiftclickError("explicit selection run_id does not match the shiftclick source")
        allowed_kinds = (
            {"grid_point", "candidate_id"}
            if spec.source_kind == "grid"
            else {"optimizer_winner"}
        )
        if selection.kind not in allowed_kinds:
            raise ShiftclickError("explicit selection kind does not match the shiftclick source kind")
        if spec.selection_kind == "candidate_id" and selection.candidate_id != spec.selection_value:
            raise ShiftclickError("explicit selection candidate_id does not match the shiftclick spec")
        if spec.selection_kind == "index" and selection.index != spec.selection_value:
            raise ShiftclickError("explicit selection index does not match the shiftclick spec")
        if spec.selection_kind == "coordinates" and selection.coordinate != spec.selection_value:
            raise ShiftclickError("explicit selection coordinate does not match the shiftclick spec")
    policy = ReplayObservationPolicy.from_spec(spec)
    run_id = f"shiftclick_{spec.id}"
    created_run_dir = False
    if output_dir is None:
        run_dir = store.allocate_run_dir("shiftclick", run_id)
        created_run_dir = True
    else:
        run_dir = Path(output_dir).resolve()
        assert_contained_path(run_dir, store.runs_dir, allow_symlinks=False)
        if run_dir.exists():
            raise FileExistsError(f"immutable shiftclick output already exists: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=False)
        created_run_dir = True
    try:
        source_manifest = store.load_manifest(selection.run_id)
        source_table = load_attested_evaluation_table(
            source_manifest,
            store=store,
            run_id=selection.run_id,
        )
        plan = normalize_selection(selection, store=store, observation_level="full_trace",
            trace_interval=spec.trace_interval, trace_actions=spec.trace_actions,
            evaluation_table=source_table)
        resolved = source_manifest.get("resolved_spec", {})
        artifact_selection = resolved.get("evaluator_artifact_selection") if isinstance(resolved, Mapping) else None
        named_runtime = resolved.get("named_runtime") if isinstance(resolved, Mapping) else None
        if artifact_selection is None and isinstance(named_runtime, Mapping):
            artifact_selection = named_runtime.get("selected_evaluator")
        artifact_path = store.get_run_dir(selection.run_id) / "evaluator_artifact"
        if not isinstance(artifact_selection, Mapping):
            raise ShiftclickError("source run has no selected evaluator provenance")
        selected = SelectedEvaluator.load(artifact_path)
        if selected.provenance != dict(artifact_selection):
            raise ShiftclickError("run-local evaluator differs from source provenance")
        if plan.pair_spec.id != spec.pair_id:
            raise ShiftclickError("shiftclick pair_id does not match the source run")
        if plan.scenario_spec.id != spec.scenario_id:
            raise ShiftclickError("shiftclick scenario_id does not match the source run")
        if plan.policy_id != spec.policy_id:
            raise ShiftclickError("shiftclick policy_id does not match the source run")
        source_row = plan.source_row
        if source_row.status != "ok":
            raise ShiftclickError(f"cannot replay non-successful source row: {source_row.status!r}")
        candidate_id, ordinal = source_row.candidate_id, source_row.ordinal
        scenarios = _source_scenarios(source_manifest, plan.scenario_spec)
        evaluation_ids = _evaluation_ids(plan, scenarios, str(source_manifest.get("run_kind")))
        staging = run_dir / ".replay_staging"
        materials, evaluations, compiled_plans = {}, [], []
        observation = {item.name: value for item in selected.compiler.schema.descriptors
            for path, value in (("evaluate_batch.metric_projection", "full"),
                ("evaluate_batch.observation.kind", "full_trace"),
                ("evaluate_batch.observation.trace_interval", spec.trace_interval),
                ("evaluate_batch.observation.trace_actions", spec.trace_actions))
            if item.lowering_path == path}
        for index, (scenario, evaluation_id) in enumerate(zip(scenarios, evaluation_ids, strict=True)):
            scenario = policy.scenario(scenario)
            material = LocalSessionMaterialization.from_scenario(scenario, repository=store.root_dir,
                manifest_root=staging / "sessions", session_id=f"{run_id}_{index:03d}").validated()
            scenario_plan = compile_selected_replay(replace(plan, scenario_spec=scenario),
                manifest=source_manifest, selected_evaluator=selected, materialization=material,
                yb_off=not policy.compare_to_source, evaluation_id=evaluation_id)
            compiled = scenario_plan.compiled_candidate
            assert compiled is not None
            evaluations.append(CompiledEvaluation.from_plan(compiled, compiler=selected.compiler,
                artifact_sha256=selected.artifact_sha256, observation=observation,
                ordinal=ordinal * len(scenarios) + index, evaluation_id=evaluation_id))
            compiled_plans.append(scenario_plan); materials[compiled.scenario_key.sha256] = material
        groups = group_evaluations(evaluations, artifact_sha256=selected.artifact_sha256,
            parameter_schema=selected.compiler.schema)
        executed = execute_local_groups(selected, groups,
            lambda group: bind_local_session_group(group, materials[group.scenario_key.sha256]),
            evaluation_ids, work_dir=staging, chunk_size=1,
            max_workers=min(len(groups), 10), artifact_dir="sidecars")
        replays = [executed.results_by_evaluation_id[value] for value in evaluation_ids]
        if any(result.status != "ok" or result.artifacts is None for result in replays):
            raise ShiftclickError("full replay did not return complete trace artifacts")
        npz_path, trace_json, _ = pack_replay_archive(run_dir, staging,
            source_run_id=selection.run_id, candidate_id=candidate_id, ordinal=ordinal,
            require_actions=spec.trace_actions,
            scenarios=[{"id": scenario.id, "evaluation_id": evaluation_id,
                "economic_fingerprint": result.economic_fingerprint, "artifacts": result.artifacts}
                for scenario, evaluation_id, result in zip(scenarios, evaluation_ids, replays, strict=True)])
        replay = replays[0]
        projection = source_table.metric_projection
        if projection is None:
            raise ShiftclickError("source evaluation table has no MetricProjection")
        projected = set(projection.fields)
        comparison_fields = tuple(
            field for field in selected.verified_evaluator.metric_fields
            if field in projected
        )
        if not comparison_fields:
            raise ShiftclickError("table projection has no evaluator metrics to compare")
        comparison, comparison_receipt = policy.compare(
            source_row.metrics,
            replay.metrics,
            expected_fingerprint=source_row.economic_fingerprint or "",
            observed_fingerprint=replay.economic_fingerprint,
            fields=comparison_fields,
        )
        replay_payload = {"candidate_id": candidate_id, "ordinal": ordinal, "status": "ok",
            "economic_fingerprint": replay.economic_fingerprint, "metrics": replay.metrics,
            "scenarios": [{"id": scenario.id, "evaluation_id": evaluation_id,
                "economic_fingerprint": result.economic_fingerprint, "metrics": result.metrics}
                for scenario, evaluation_id, result in zip(scenarios, evaluation_ids, replays, strict=True)]}
        atomic_write_json(run_dir / "replay_result.json", replay_payload)
        atomic_write_json(run_dir / "economic_comparison.json", comparison_receipt)
        artifacts = [_artifact(npz_path, run_dir, "replay_trace_npz"),
            _artifact(trace_json, run_dir, "replay_trace_companion")]
        for path, kind in ((run_dir / "replay_result.json", "replay_result"), (run_dir / "economic_comparison.json", "economic_comparison")):
            artifacts.append({"path": path.relative_to(run_dir).as_posix(), "kind": kind, "bytes": path.stat().st_size, "sha256": sha256_path(path)})
        manifest = new_shiftclick_manifest(
            run_id=run_id,
            shiftclick_id=spec.id,
            source_run_id=selection.run_id,
            selection=selection.to_dict(),
            resolution="full",
            resolved_spec={
                "shiftclick": spec.to_dict(),
                "observation_policy": policy.to_dict(),
                "replay_plan": replace(plan, artifact_dir=None).to_dict(),
                "metric_projection": projection.to_dict(),
                "evaluator_source": {"source_run_id": selection.run_id,
                    "selected_evaluator": selected.provenance},
            },
            execution={"scope": "local"},
            core=selected.manifest_core(
                binary_override=f"source_run/{selection.run_id}/evaluator_artifact/evaluator"),
            artifacts=artifacts,
        )
        write_manifest_atomic(run_dir / "manifest.json", manifest, expected_kind="shiftclick")
        return ShiftclickResult(
            run_dir,
            replace(compiled_plans[0], artifact_dir=None),
            candidate_id,
            replay_payload,
            comparison,
            comparison_receipt,
            manifest,
        )
    except BaseException:
        if created_run_dir and not (run_dir / "manifest.json").is_file():
            assert_contained_path(run_dir, store.runs_dir, allow_symlinks=False)
            if run_dir.is_dir():
                shutil.rmtree(run_dir)
        raise




__all__ = [
    "ReplayObservationPolicy",
    "ShiftclickError",
    "ShiftclickResult",
    "run_shiftclick",
    "selection_from_spec",
]
