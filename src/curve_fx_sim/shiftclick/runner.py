"""Strict SelectionRef -> ReplayPlan shiftclick execution and verification."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..analysis.economics import EconomicComparison, compare_economics
from ..artifacts.io import atomic_write_json, sha256_path
from ..artifacts.manifest import new_shiftclick_manifest, write_manifest_atomic
from ..artifacts.store import RunStore
from ..evaluation.client import HarnessClient
from ..evaluation.identity import VerifiedEvaluator, validate_evaluator_identity
from curve_fx_harness_client.models import CandidateSpec, ObservationSpec
from ..evaluation.selection import (
    ReplayPlan,
    SelectionRef,
    load_attested_evaluation_table,
    normalize_selection,
)
from ..specs.shiftclick import ShiftclickSpec
from ..specs.common import SpecError, assert_contained_path, repository_relative
from ..plotting.trajectory import load_trajectory


class ShiftclickError(ValueError):
    """Raised when a shiftclick source or full replay fails closed."""


@dataclass(frozen=True)
class ShiftclickResult:
    run_dir: Path
    plan: ReplayPlan
    source_candidate_id: str
    replay_result: Mapping[str, Any]
    comparison: EconomicComparison
    manifest: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_dir": self.run_dir.as_posix(),
            "plan": self.plan.to_dict(),
            "source_candidate_id": self.source_candidate_id,
            "replay_result": dict(self.replay_result),
            "comparison": self.comparison.to_dict(),
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


def _artifact_path(raw: str, *, run_dir: Path, root: Path, label: str) -> Path:
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = (root / candidate).resolve()
        if not candidate.is_file():
            candidate = (run_dir / raw).resolve()
    else:
        candidate = candidate.resolve()
    try:
        candidate.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise ShiftclickError(f"{label} escapes shiftclick run directory: {raw!r}") from exc
    if not candidate.is_file():
        raise ShiftclickError(f"{label} artifact is missing: {raw!r}")
    return candidate


def _result_artifacts(
    result: Any,
    *,
    run_dir: Path,
    root: Path,
    require_actions: bool = False,
) -> tuple[dict[str, Any], ...]:
    artifacts = result.artifacts
    if artifacts is None or not artifacts.trace_path:
        raise ShiftclickError("full replay returned no trace artifact")
    descriptors: list[dict[str, Any]] = []
    for label, raw, expected in (
        ("trace", artifacts.trace_path, artifacts.trace_sha256),
        ("actions", artifacts.actions_path, artifacts.actions_sha256),
    ):
        if not raw:
            if label == "trace" or require_actions:
                raise ShiftclickError(f"full replay returned no {label} artifact")
            continue
        path = _artifact_path(raw, run_dir=run_dir, root=root, label=label)
        digest = sha256_path(path)
        if expected and digest.lower() != expected.lower():
            raise ShiftclickError(f"{label} artifact hash mismatch")
        item = {
            "path": path.relative_to(run_dir).as_posix(),
            "kind": label,
            "bytes": path.stat().st_size,
            "sha256": digest,
        }
        if label == "trace":
            load_trajectory(path)
        descriptors.append(item)
    return tuple(descriptors)
def run_shiftclick(
    spec: ShiftclickSpec,
    *,
    store: RunStore,
    client: HarnessClient,
    pair_spec: Any | None = None,
    scenario_spec: Any | None = None,
    output_dir: Path | None = None,
) -> ShiftclickResult:
    """Replay exactly one normalized source candidate with full observation."""
    selection = selection_from_spec(spec)
    run_id = f"shiftclick_{spec.id}"
    if output_dir is None:
        run_dir = store.allocate_run_dir("shiftclick", run_id)
    else:
        run_dir = Path(output_dir).resolve()
        assert_contained_path(run_dir, store.runs_dir, allow_symlinks=False)
        if run_dir.exists():
            raise FileExistsError(f"immutable shiftclick output already exists: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=False)
    try:
        source_manifest = store.load_manifest(selection.run_id)
        source_table = load_attested_evaluation_table(
            source_manifest,
            store=store,
            run_id=selection.run_id,
        )
        plan = normalize_selection(
            selection,
            store=store,
            pair_spec=pair_spec,
            scenario_spec=scenario_spec,
            observation_level="full_trace",
            trace_interval=spec.trace_interval,
            trace_actions=spec.trace_actions,
            artifact_dir=run_dir / "trace",
            evaluation_table=source_table,
        )
        if plan.pair_spec.id != spec.pair_id:
            raise ShiftclickError("shiftclick pair_id does not match the source run")
        if plan.scenario_spec.id != spec.scenario_id:
            raise ShiftclickError("shiftclick scenario_id does not match the source run")
        if plan.policy_id != spec.policy_id:
            raise ShiftclickError("shiftclick policy_id does not match the source run")
        source_row = plan.source_row
        if source_row.status != "ok":
            raise ShiftclickError(f"cannot replay non-successful source row: {source_row.status!r}")
        candidate_id = source_row.candidate_id
        ordinal = source_row.ordinal
        policy_params = plan.policy_params
        if isinstance(policy_params, Mapping) and "vector" in policy_params:
            policy_params = policy_params["vector"]
        request = CandidateSpec(
            ordinal=ordinal,
            candidate_id=candidate_id,
            policy_params=copy.deepcopy(policy_params),
            pool_overrides=copy.deepcopy(plan.pool_overrides),
        )
        identity = client.prepare()
        if not isinstance(identity, VerifiedEvaluator):
            raise TypeError("HarnessClient.prepare() must return a VerifiedEvaluator")
        source_core = source_manifest.get("core", {})
        expected_policy_id = str(source_core.get("policy_id") or plan.policy_id)
        if not expected_policy_id or plan.policy_id != expected_policy_id:
            raise ShiftclickError("source manifest and replay plan policy identities disagree")
        if spec.policy_id != expected_policy_id:
            raise ShiftclickError("shiftclick policy_id does not match the selected source run")
        validate_evaluator_identity(
            identity,
            expected_policy_id=expected_policy_id,
            expected_policy_source_sha256=source_core.get("policy_source_sha256"),
            expected_policy_abi=source_core.get("policy_abi"),
            expected_policy_parameter_count=source_core.get("policy_parameter_count"),
        )
        client.open_session(plan.scenario_spec, session_id=run_id)
        response = client.evaluate_batch(
            (request,),
            observation=ObservationSpec(
                kind="full_trace",
                trace_interval=plan.trace_interval,
                trace_actions=plan.trace_actions,
                artifact_dir=repository_relative(run_dir / "trace", root=store.root_dir).as_posix(),
            ),
        )
        if response.status != "complete" or len(response.results) != 1:
            raise ShiftclickError("full replay did not return exactly one complete result")
        replay = response.results[0]
        if replay.candidate_id != candidate_id or replay.ordinal != ordinal:
            raise ShiftclickError("full replay identity does not match source candidate")
        artifact_descriptors = _result_artifacts(
            replay,
            run_dir=run_dir,
            root=store.root_dir,
            require_actions=plan.trace_actions,
        )
        projection = source_table.metric_projection
        if projection is None:
            raise ShiftclickError("source evaluation table has no MetricProjection")
        comparison = compare_economics(
            source_row.metrics,
            replay.metrics,
            expected_fingerprint=source_row.economic_fingerprint or "",
            observed_fingerprint=replay.economic_fingerprint,
            fields=projection.fields,
        )
        replay_payload = replay.model_dump()
        atomic_write_json(run_dir / "replay_result.json", replay_payload)
        atomic_write_json(run_dir / "economic_comparison.json", comparison.to_dict())
        artifacts = list(artifact_descriptors)
        for path, kind in ((run_dir / "replay_result.json", "replay_result"), (run_dir / "economic_comparison.json", "economic_comparison")):
            artifacts.append({"path": path.relative_to(run_dir).as_posix(), "kind": kind, "bytes": path.stat().st_size, "sha256": sha256_path(path)})
        manifest = new_shiftclick_manifest(
            run_id=run_id,
            shiftclick_id=spec.id,
            source_run_id=selection.run_id,
            selection=selection.to_dict(),
            resolution="full",
            resolved_spec={"shiftclick": spec.to_dict(), "replay_plan": plan.to_dict(), "metric_projection": projection.to_dict()},
            execution={"scope": "local"},
            core=identity.to_core_dict(),
            artifacts=artifacts,
        )
        write_manifest_atomic(run_dir / "manifest.json", manifest, expected_kind="shiftclick")
        return ShiftclickResult(
            run_dir,
            plan,
            candidate_id,
            replay_payload,
            comparison,
            manifest,
        )
    finally:
        client.close()




__all__ = ["ShiftclickError", "ShiftclickResult", "run_shiftclick", "selection_from_spec"]
