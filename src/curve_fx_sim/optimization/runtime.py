"""Complete local and distributed optimization execution runtime with full immutable replay inputs & lineage."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shlex
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..artifacts.attestation import (
    find_attested_artifact,
    load_attested_evaluation_table,
)
from ..artifacts.io import atomic_write_json, canonical_json_bytes, sha256_path
from ..artifacts.manifest import new_optimization_manifest
from ..artifacts.store import RunStore
from ..artifacts.tables import EvaluationRow, EvaluationTable, MetricProjection
from ..evaluation.grouping import (
    CompiledEvaluation,
    bind_local_session_group,
    group_evaluations,
)
from ..evaluation.selected import SelectedEvaluator, materialize_selected_evaluator
from ..evaluation.session import LocalSessionMaterialization
from ..evaluation.selection import SelectionRef
from ..execution.adapter import SSHProcessAdapter
from ..execution.grouped import execute_local_groups
from ..execution.grouped_dispatch import dispatch_grouped_evaluations
from ..execution.site import SiteProfile, load_site_profile
from ..execution.shared_nfs import (
    canonical_sha256,
    execution_site_payload,
    package_identity_sha256,
    shared_run_lease,
)
from ..specs.common import ProjectContext
from ..specs.optimization import OptimizationSpec, load_optimization_spec
from ..specs.pair import load_pair_spec
from ..specs.scenario import load_scenario_spec
from .nevergrad_adapter import NevergradTwoPointsDEOptimizer
from .profiles import NamedProfile
from .requests import compile_named_request
from .scoring import (
    SCORE_FX_LP_E15_SLIPPAGE_V1_KEY,
    SCORE_VERSION,
    loss_from_score,
    objective_failure_count,
    normalize_score_key,
    score_objective_value,
    score_scenarios,
)
from .search import SearchLayout, select_search_descriptors
from .tmrbcd import TmrbcdOptimizer

OPTIMIZATION_CHECKPOINT_SCHEMA_VERSION = 6


def _row_prefix_sha256(rows: Sequence[EvaluationRow], step: int) -> str:
    return hashlib.sha256(
        canonical_json_bytes([row.to_dict() for row in rows[:step]])
    ).hexdigest()


def _compile_named_batch(
    profile: NamedProfile,
    selected: SelectedEvaluator,
    scenarios: Sequence[Any],
    materializations: Mapping[str, LocalSessionMaterialization],
    vectors: Sequence[Sequence[int | float]],
    *,
    start: int,
) -> tuple[list[dict[str, Any]], tuple[Any, ...]]:
    """Compile one ask batch and bind its portable evaluation lineage."""
    observation = {
        item.name: "summary"
        for item in selected.compiler.schema.descriptors
        if item.lowering_path == "evaluate_batch.metric_projection"
    }
    compiled: list[dict[str, Any]] = []
    evaluations: list[CompiledEvaluation] = []
    for ask_index, raw_vector in enumerate(vectors):
        proposal = profile.to_proposal(raw_vector)
        vector = [proposal[item.name] for item in profile.layout.dimensions]
        plans, ask_evaluations = {}, {}
        for scenario_index, scenario in enumerate(scenarios):
            materialization = materializations[scenario.id]
            plan = compile_named_request(
                profile,
                vector,
                selected.compiler,
                open_session=materialization.baseline_open_session_fields,
                scenario=materialization.closure,
            )
            evaluation_id = f"c_{start + ask_index:05d}_{scenario.id}"
            evaluation = CompiledEvaluation.from_plan(
                plan,
                compiler=selected.compiler,
                artifact_sha256=selected.artifact_sha256,
                observation=observation,
                ordinal=(start + ask_index) * len(scenarios) + scenario_index,
                evaluation_id=evaluation_id,
            )
            plans[scenario.id] = plan
            ask_evaluations[scenario.id] = evaluation
            evaluations.append(evaluation)
        candidate_payloads = {plan.candidate_json for plan in plans.values()}
        if len(candidate_payloads) != 1:
            raise ValueError("candidate lowering differs across scenario sessions")
        compiled.append({"vector": vector, "plans": plans, "evaluations": ask_evaluations})

    groups = group_evaluations(
        evaluations,
        artifact_sha256=selected.artifact_sha256,
        parameter_schema=selected.compiler.schema,
    )
    group_by_evaluation = {
        evaluation.evaluation_id: group.key.sha256
        for group in groups
        for evaluation in group.evaluations
    }
    for ask in compiled:
        ask["evaluation_lineage"] = [
            {
                "evaluation_id": evaluation.evaluation_id,
                "session_group_id": group_by_evaluation[evaluation.evaluation_id],
                "observation_id": evaluation.observation_key.sha256,
                "candidate_sha256": evaluation.candidate.candidate_sha256,
                "scenario_id": scenario.id,
            }
            for scenario in scenarios
            for evaluation in (ask["evaluations"][scenario.id],)
        ]
    return compiled, groups


def _inspect_remote_execution_closure(site: SiteProfile) -> dict[str, str]:
    repository = str(site.cluster.repository_root)
    worker = shlex.quote(site.cluster.worker_command)
    command = (
        f"sha256sum {worker} && "
        f"{worker} --project-root {shlex.quote(repository)} worker package-identity"
    )
    result = SSHProcessAdapter(ssh_config=site.ssh).run_ssh(
        site.cluster.coordinator, command
    )
    if not result.ok:
        raise RuntimeError(
            f"failed to inspect remote execution closure on {site.cluster.coordinator}: {result.stderr}"
        )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 2:
        raise RuntimeError("remote execution closure inspection returned invalid output")
    return {
        "worker_sha256": lines[0].split(maxsplit=1)[0],
        "package_sha256": lines[1],
    }


@dataclass
class OptimizationStatus:
    """Status snapshot of an ongoing or settled optimization run."""

    run_id: str
    status: str
    candidates_evaluated: int
    best_loss: float
    best_objective: float
    best_params: list[float]
    elapsed_seconds: float
    best_score: dict[str, Any] | None = None
    step: int = 0
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "candidates_evaluated": self.candidates_evaluated,
            "best_loss": self.best_loss,
            "best_objective": self.best_objective,
            "best_params": self.best_params,
            "elapsed_seconds": self.elapsed_seconds,
            "best_score": self.best_score,
            "step": self.step,
            "updated_at": self.updated_at,
        }


@dataclass
class OptimizationResult:
    """Final artifact bundle produced by an optimization run."""

    run_id: str
    spec: OptimizationSpec
    table: EvaluationTable
    winner: SelectionRef
    top_k: list[SelectionRef]
    manifest_path: Path
    table_path: Path
    winner_path: Path
    topk_path: Path
    best_score: dict[str, Any]
    candidates_evaluated: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "spec": self.spec.to_dict(),
            "winner": self.winner.to_dict(),
            "top_k": [s.to_dict() for s in self.top_k],
            "best_score": self.best_score,
            "candidates_evaluated": self.candidates_evaluated,
            "manifest_path": self.manifest_path.as_posix(),
            "table_path": self.table_path.as_posix(),
            "winner_path": self.winner_path.as_posix(),
            "topk_path": self.topk_path.as_posix(),
        }


def _execution_context(
    repository: Path | None,
    store: RunStore | None,
) -> ProjectContext:
    if store is not None:
        if repository is not None and repository.resolve() != store.context.project_root:
            raise ValueError("repository does not match the RunStore project root")
        return store.context
    if repository is None:
        raise ValueError("optimization requires an explicit repository or RunStore")
    return ProjectContext.from_root(repository)


def _run_optimization(
    spec_or_path: str | os.PathLike[str] | OptimizationSpec,
    *,
    store: RunStore | None = None,
    run_id: str | None = None,
    resume: bool = False,
    budget: int | None = None,
    batch_size: int | None = None,
    top_k_count: int = 8,
    repository: Path | None = None,
    site: str = "local",
    blades: Sequence[str] = (),
    selected_evaluator: SelectedEvaluator,
) -> OptimizationResult:
    context = _execution_context(repository, store)
    root = context.project_root
    run_store = store if store is not None else RunStore(context)
    distributed = site != "local" or bool(blades)
    site_profile = load_site_profile(site, root=root) if distributed else None
    target_blades = (
        list(blades) or list(site_profile.cluster.blades)
        if site_profile is not None
        else []
    )
    if distributed and not target_blades:
        raise ValueError(f"site {site_profile.name!r} has no blades configured")
    if site_profile is not None:
        site_profile.validate_blades(target_blades)
    if distributed:
        assert site_profile is not None
        if (
            site_profile.cluster.transport != "shared_nfs"
            or root != Path(str(site_profile.cluster.repository_root)).resolve()
            or run_store.runs_dir.resolve()
            != Path(str(site_profile.cluster.remote_run_root)).resolve()
        ):
            raise ValueError(
                "artifact-selected distributed optimization must be launched on the configured "
                "coordinator with its shared repository and remote run root"
            )

    if isinstance(spec_or_path, OptimizationSpec):
        opt_spec = spec_or_path
    else:
        opt_spec = load_optimization_spec(spec_or_path, repository=root)

    actual_run_id = run_id or opt_spec.id or f"opt_{uuid.uuid4().hex[:12]}"

    total_budget = int(
        budget if budget is not None else opt_spec.optimizer_config.get("budget", 100)
    )
    eval_batch_size = int(
        batch_size if batch_size is not None else opt_spec.optimizer_config.get("batch_size", 4)
    )
    if total_budget <= 0:
        raise ValueError(f"optimization budget must be positive, got {total_budget}")
    if eval_batch_size <= 0:
        raise ValueError(f"optimization batch_size must be positive, got {eval_batch_size}")
    if top_k_count <= 0:
        raise ValueError(f"top_k_count must be positive, got {top_k_count}")
    pair_spec = load_pair_spec(opt_spec.pair_id, repository=root)
    scenarios = [load_scenario_spec(s, repository=root) for s in opt_spec.scenarios]
    if not scenarios:
        raise ValueError(f"Optimization spec {opt_spec.id} has no scenarios configured")
    primary_scenario = scenarios[0]
    template_json = None
    if primary_scenario.template_path is not None:
        template_file = root / primary_scenario.template_path
        if template_file.is_file():
            with template_file.open("r", encoding="utf-8") as _template_stream:
                template_json = json.load(_template_stream)

    run_dir: Path | None = None
    named_scenarios: dict[str, LocalSessionMaterialization] = {}
    select_search_descriptors(selected_evaluator.compiler.schema, opt_spec.parameter_space)
    selected_policy = selected_evaluator.policy_identity
    if opt_spec.policy_id != selected_policy["id"]:
        raise ValueError(f"optimization policy_id {opt_spec.policy_id!r} does not match selected evaluator policy {selected_policy['id']!r}")
    run_dir = run_store.get_run_dir(actual_run_id) if resume else run_store.allocate_run_dir("optimization", actual_run_id)
    selected_evaluator = materialize_selected_evaluator(selected_evaluator, run_dir, resume=resume)
    named_scenarios = {s.id: LocalSessionMaterialization.from_scenario(s, repository=root, manifest_root=run_dir / "session_transport", session_id=f"opt_{actual_run_id}_{s.id}") for s in scenarios}
    primary_materialization = named_scenarios[primary_scenario.id]
    layout = SearchLayout.from_schema(selected_evaluator.compiler.schema, opt_spec.parameter_space, template_json, primary_materialization.baseline_open_session_fields)
    profile = NamedProfile(opt_spec.algorithm.lower(), layout)
    evaluator_ident_dict = selected_evaluator.manifest_core(binary_override="evaluator_artifact/evaluator")
    evaluator_metric_fields = tuple(str(field) for field in evaluator_ident_dict.get("metric_fields", ()))
    if not evaluator_metric_fields:
        raise ValueError("evaluator identity must attest non-empty metric_fields")
    evaluator_metric_fields = tuple({*evaluator_metric_fields, "loss", "objective_value", SCORE_FX_LP_E15_SLIPPAGE_V1_KEY, "gate", "mean_e15_penalty", "mean_slippage_penalty", "mean_detach_energy_ungated", "mean_tw_real_slippage_1pct", "n_scenarios", "n_failed", "n_yb_failed", "objective_failures"})
    metric_projection = MetricProjection.from_fields(
        evaluator_metric_fields,
        projection_id="optimization-primary-scenario",
    )

    policy_header = "artifact:policy_header"; verified_policy_sha256 = selected_policy["source_sha256"]; policy_abi = selected_policy["abi"]
    policy_parameter_names = tuple(d.name for d in sorted((d for d in selected_evaluator.compiler.schema.descriptors if d.order is not None), key=lambda d: d.order))
    # Resolve scenario templates and hashes
    resolved_scenarios_info: list[dict[str, Any]] = []
    for sc in scenarios:
        tmpl_path = str(sc.template_path.as_posix()) if sc.template_path else ""
        tmpl_sha = sc.template_sha256 or ""
        if tmpl_path:
            full_tmpl_p = root / sc.template_path
            if not full_tmpl_p.is_file():
                raise FileNotFoundError(f"scenario template not found: {full_tmpl_p}")
            calculated_template_sha = sha256_path(full_tmpl_p)
            if tmpl_sha and tmpl_sha != calculated_template_sha:
                raise ValueError(
                    f"scenario template SHA-256 mismatch for {sc.id}: "
                    f"expected {tmpl_sha}, calculated {calculated_template_sha}"
                )
            tmpl_sha = calculated_template_sha
        for market_file in sc.market_files:
            full_market_path = root / market_file.path
            if not full_market_path.is_file():
                raise FileNotFoundError(f"scenario market file not found: {full_market_path}")
            calculated_market_sha = sha256_path(full_market_path)
            if market_file.sha256 and market_file.sha256 != calculated_market_sha:
                raise ValueError(
                    f"scenario market SHA-256 mismatch for {market_file.path}: "
                    f"expected {market_file.sha256}, calculated {calculated_market_sha}"
                )

        resolved_scenario = sc.to_dict()
        resolved_scenario["template_path"] = tmpl_path
        resolved_scenario["template_sha256"] = tmpl_sha
        resolved_scenarios_info.append(resolved_scenario)

    # YB and Scoring Configuration
    scoring_cfg = opt_spec.scoring_config
    misplaced_yb = sorted(
        set(scoring_cfg).intersection({"yb_fee", "yb_stride"})
    )
    if misplaced_yb:
        raise ValueError(
            "YB economics belong to typed ScenarioSpec fields, not scoring_config: "
            + ", ".join(misplaced_yb)
        )
    score_key = normalize_score_key(
        str(scoring_cfg.get("score_key", SCORE_FX_LP_E15_SLIPPAGE_V1_KEY))
    )
    require_yb = bool(scoring_cfg.get("require_yb", False))
    if require_yb and any(not scenario.yb_releverage for scenario in scenarios):
        raise ValueError(
            "require_yb=true requires yb_releverage=true in every ScenarioSpec"
        )
    yb_settings = {
        "require_yb": require_yb,
        "scenarios": {
            scenario.id: {
                key: value
                for key, value in scenario.harness_session_config().items()
                if key.startswith("yb_")
            }
            for scenario in scenarios
        },
        "score_version": SCORE_VERSION,
        "score_key": score_key,
    }

    # 2. Instantiate the selected optimizer over the resolved exact lattice.
    algorithm = opt_spec.algorithm.lower()
    seed = int(opt_spec.optimizer_config.get("seed", 42))
    if algorithm == "tmrbcd":
        optimizer = TmrbcdOptimizer(
            lattice=profile.lattice,
            initial_params=profile.initial_vector,
            budget=total_budget,
            seed=seed,
        )
    elif algorithm == "nevergrad_two_points_de":
        optimizer = NevergradTwoPointsDEOptimizer(
            lattice=profile.lattice,
            initial_params=profile.initial_vector,
            budget=total_budget,
            seed=seed,
            num_workers=eval_batch_size,
        )
    else:
        raise ValueError(f"unsupported optimization algorithm {algorithm!r}")

    session_receipts = {s.id: {"scenario_key": named_scenarios[s.id].scenario_key.sha256, "baseline_request_sha256": named_scenarios[s.id].baseline_request_sha256} for s in scenarios}
    named_runtime_identity = {"selected_evaluator": selected_evaluator.provenance, "search_geometry": profile.geometry_receipt, "scenario_sessions": session_receipts}
    run_identity = {
        "optimization": opt_spec.to_dict(),
        "policy": selected_evaluator.policy_identity,
        "profile": profile.geometry_receipt,
        "pair": pair_spec.to_dict(),
        "scenarios": resolved_scenarios_info,
        "evaluator": evaluator_ident_dict,
        "score_key": score_key,
        "named_runtime": named_runtime_identity,
    }
    run_identity_sha256 = hashlib.sha256(canonical_json_bytes(run_identity)).hexdigest()
    execution_closure_sha256: str | None = None
    if distributed and site_profile is not None and site_profile.cluster.transport == "shared_nfs":
        remote_execution = _inspect_remote_execution_closure(site_profile)
        local_package_sha256 = package_identity_sha256(root)
        if remote_execution["package_sha256"] != local_package_sha256:
            raise ValueError("remote optimizer package differs from the local execution closure")
        execution_closure_sha256 = canonical_sha256(
            {
                "run_identity": run_identity,
                "package_sha256": local_package_sha256,
                "worker_sha256": remote_execution["worker_sha256"],
                "site": execution_site_payload(site_profile, target_blades),
            }
        )
        expected_closure = os.environ.get("FXSIM_EXECUTION_CLOSURE_SHA256")
        if expected_closure and expected_closure != execution_closure_sha256:
            raise ValueError("remote resolved execution closure differs from coordinator request")
    # 3. The table is the durable ask-order prefix; the checkpoint only caches optimizer state.
    immutable_table_metadata = {
        "run_id": actual_run_id,
        "optimization_id": opt_spec.id,
        "algorithm": algorithm,
        "policy_id": opt_spec.policy_id,
        "policy_header": policy_header,
        "policy_source_sha256": verified_policy_sha256,
        "policy_abi": policy_abi,
        "parameter_names": list(policy_parameter_names),
        "lattice": profile.geometry_receipt,
        "pair_id": opt_spec.pair_id,
        "pair": pair_spec.to_dict(),
        "scenario": primary_scenario.to_dict(),
        "scenarios": resolved_scenarios_info,
        "evaluator_identity": evaluator_ident_dict,
        "yb_settings": yb_settings,
        "score_key": score_key,
        "score_version": SCORE_VERSION,
        "execution_closure_sha256": execution_closure_sha256,
        "named_runtime": named_runtime_identity,
        "run_identity_sha256": run_identity_sha256,
    }
    checkpoint_file = run_dir / "checkpoint.json"
    table_path = run_dir / "evaluation_table.npz"
    rows: list[EvaluationRow] = []
    table: EvaluationTable | None = None
    checkpoint_step = 0

    if resume:
        if table_path.is_file():
            table = EvaluationTable.from_npz(table_path)
            rows = table.rows
            if table.metric_projection != metric_projection:
                raise ValueError("Resume evaluation table metric projection changed")
            if table.metadata.get("run_identity_sha256") != run_identity_sha256:
                raise ValueError("Resume evaluation table identity changed")
            if table.metadata.get("candidates_evaluated") != len(rows):
                raise ValueError("Resume evaluation table row count metadata is invalid")
            if any(left.ordinal >= right.ordinal for left, right in zip(rows, rows[1:])):
                raise ValueError("Resume evaluation table ordinals must be strictly increasing")
        if checkpoint_file.is_file():
            with checkpoint_file.open("r", encoding="utf-8") as f:
                cp_data = json.load(f)
            if cp_data.get("schema_version") != OPTIMIZATION_CHECKPOINT_SCHEMA_VERSION:
                raise ValueError("Resume checkpoint schema is stale; start a new immutable run")
            if cp_data.get("run_identity_sha256") != run_identity_sha256:
                raise ValueError("Resume identity mismatch: policy, lattice, scenario, or evaluator changed")
            if cp_data.get("execution_closure_sha256") != execution_closure_sha256:
                raise ValueError("Resume identity mismatch: execution closure changed")
            checkpoint_step = cp_data.get("step")
            if isinstance(checkpoint_step, bool) or not isinstance(checkpoint_step, int) or checkpoint_step < 0:
                raise ValueError("Resume checkpoint step must be a non-negative integer")
            if checkpoint_step > len(rows):
                raise ValueError("Resume checkpoint is ahead of the evaluation table")
            if cp_data.get("row_prefix_sha256") != _row_prefix_sha256(rows, checkpoint_step):
                raise ValueError("Resume checkpoint row prefix changed")
            opt_state = cp_data.get("optimizer_state")
            if not isinstance(opt_state, Mapping):
                raise ValueError("Resume checkpoint optimizer_state must be an object")
            optimizer.restore(opt_state)
            if optimizer.step != checkpoint_step:
                raise ValueError("Resume checkpoint optimizer step is inconsistent")
        if rows:
            recompiled, _ = _compile_named_batch(profile, selected_evaluator, scenarios,
                named_scenarios, [row.params["vector"] for row in rows], start=0)
            if [item["evaluation_lineage"] for item in recompiled] != [row.params.get("evaluation_lineage") for row in rows]:
                raise ValueError("Resume identity mismatch: evaluation group lineage changed")
        for start in range(checkpoint_step, len(rows), eval_batch_size):
            chunk = rows[start : start + eval_batch_size]
            regenerated = optimizer.ask(len(chunk))
            reproduced = [[profile.to_proposal(vector)[item.name]
                           for item in profile.layout.dimensions] for vector in regenerated]
            if reproduced != [row.params["vector"] for row in chunk]:
                raise ValueError("Resume identity mismatch: optimizer ask prefix changed")
            optimizer.tell([row.params["vector"] for row in chunk],
                [float(row.metrics["loss"]) for row in chunk],
                objectives=[float(row.metrics["objective_value"]) for row in chunk],
                scores=[dict(row.metrics) for row in chunk])

    if (resume and len(rows) >= total_budget and all((run_dir / name).is_file()
            for name in ("manifest.json", "evaluation_table.npz", "winner.json", "topk.json"))):
        return collect_optimization(actual_run_id, store=run_store, repository=root)

    step = len(rows)

    # Publish optimizer state only after the matching table prefix is durable.
    def _save_checkpoint() -> None:
        opt_snapshot = optimizer.snapshot()
        cp_payload = {
            "schema_version": OPTIMIZATION_CHECKPOINT_SCHEMA_VERSION,
            "run_id": actual_run_id,
            "optimization_id": opt_spec.id,
            "policy_id": opt_spec.policy_id,
            "policy_source_sha256": verified_policy_sha256,
            "pair_id": opt_spec.pair_id,
            "pair": pair_spec.to_dict(),
            "scenario": primary_scenario.to_dict(),
            "scenarios": resolved_scenarios_info,
            "evaluator_identity": evaluator_ident_dict,
            "yb_settings": yb_settings,
            "score_key": score_key,
            "run_identity_sha256": run_identity_sha256,
            "execution_closure_sha256": execution_closure_sha256,
            "named_runtime": named_runtime_identity,
            "step": step,
            "optimizer_state": opt_snapshot,
            "row_prefix_sha256": _row_prefix_sha256(rows, step),
        }
        atomic_write_json(checkpoint_file, cp_payload)

    if checkpoint_step < len(rows):
        _save_checkpoint()

    # 4. Optimization Loop
    while step < total_budget:
        remaining_budget = total_budget - step
        if remaining_budget <= 0:
            break

        prepared_candidates: list[dict[str, Any]] = []
        raw_asks = optimizer.ask(min(eval_batch_size, remaining_budget))
        if not raw_asks:
            break
        named_compiled, named_groups = _compile_named_batch(profile, selected_evaluator, scenarios,
                                                             named_scenarios, raw_asks, start=step)
        for cand_idx, raw_params in enumerate(raw_asks):
            compiled = named_compiled[cand_idx]; quant_params = compiled["vector"]; plans = compiled["plans"]; primary_plan = plans[primary_scenario.id]
            pool_overrides = primary_plan.pool_overrides; named_values = dict(primary_plan.named_values); candidate_sha256 = primary_plan.candidate_sha256; evaluations = compiled["evaluations"]; evaluation_lineage = compiled["evaluation_lineage"]
            cand_id = f"c_{step + cand_idx:05d}"
            ask_id = f"ask_{step + cand_idx:06d}"
            prepared_candidates.append(
                {
                    "cand_id": cand_id,
                    "cand_idx": cand_idx,
                    "ask_id": ask_id,
                    "quant_params": quant_params,
                    "pool_overrides": pool_overrides,
                    "named_values": named_values,
                    "candidate_sha256": candidate_sha256,
                    "plans": plans,
                    "evaluations": evaluations,
                    "evaluation_lineage": evaluation_lineage,
                    "lineage": {
                        "algorithm": algorithm,
                        "lane": "main",
                        "ask_id": ask_id,
                        "source": f"{algorithm}:main",
                        "step": step + cand_idx,
                        "ordinal": step + cand_idx,
                    },
                    "scenario_results": [],
                    "scenario_candidate_ids": {},
                    "scenario_fingerprints": {},
                    "primary_metrics": {},
                    "evaluation_ok": True,
                }
            )

        evaluations = tuple(c["evaluations"][scenario.id]
                            for c in prepared_candidates for scenario in scenarios)
        ordered_evaluation_ids = tuple(item.evaluation_id for item in evaluations)
        if distributed:
                assert site_profile is not None
                assignments = {blade: [] for blade in target_blades}
                for index, candidate in enumerate(prepared_candidates):
                    blade = target_blades[index % len(target_blades)]
                    assignments[blade].extend(candidate["evaluations"][scenario.id].evaluation_id
                                              for scenario in scenarios)
                execution = dispatch_grouped_evaluations(
                    run_root=run_dir, run_id=actual_run_id, selected=selected_evaluator,
                    evaluations=evaluations, scenarios=scenarios,
                    evaluation_ids_by_blade=assignments,
                    repository=root, site=site_profile, chunk_size=eval_batch_size,
                    evaluator_workers=site_profile.runner.worker_concurrency,
                    request_namespace=f"opt_{step:06d}",
                    ssh=SSHProcessAdapter(ssh_config=site_profile.ssh),
                )
        else:
            materializations = {value.scenario_key.sha256: value
                                for value in named_scenarios.values()}
            local_site = load_site_profile("local", root=root)
            execution = execute_local_groups(selected_evaluator, named_groups,
                lambda group: bind_local_session_group(group, materializations[group.scenario_key.sha256]),
                ordered_evaluation_ids, work_dir=run_dir, chunk_size=eval_batch_size,
                max_workers=local_site.runner.max_workers,
                evaluator_workers=local_site.runner.worker_concurrency)
        for c in prepared_candidates:
            for scenario in scenarios:
                evaluation = c["evaluations"][scenario.id]; assert evaluation.evaluation_id is not None
                c["scenario_candidate_ids"][scenario.id] = evaluation.evaluation_id
                res = execution.results_by_evaluation_id[evaluation.evaluation_id]
                c["evaluation_ok"] = c["evaluation_ok"] and res.status == "ok"
                c["scenario_results"].append({**res.metrics, "ok": res.status == "ok"})
                if scenario.id == primary_scenario.id:
                    c["primary_metrics"] = dict(res.metrics)
                if res.economic_fingerprint:
                    c["scenario_fingerprints"][scenario.id] = res.economic_fingerprint

        for c in prepared_candidates:
            score_res = score_scenarios(c["scenario_results"], require_yb=require_yb)
            obj_val = score_objective_value(score_res, score_key)
            score_res.update(objective_value=obj_val, objective_failures=objective_failure_count(score_res, score_key))
            c.update(loss=loss_from_score(score_res), objective=obj_val, score_res=score_res)

        # Build canonical evaluation rows from either local scoring or attested worker results.
        batch_rows: list[EvaluationRow] = []
        for c in prepared_candidates:
            score_res = c["score_res"]
            loss_val = c["loss"]
            obj_val = c["objective"]
            primary_fingerprint = c["scenario_fingerprints"].get(primary_scenario.id)
            primary_scen_cand_id = c["scenario_candidate_ids"].get(primary_scenario.id, c["cand_id"])
            row_metrics = {name: float(value) for source in (c["primary_metrics"], score_res,
                {"loss": loss_val, "objective_value": obj_val}) for name, value in source.items()
                if isinstance(value, (bool, int, float)) and math.isfinite(float(value))}
            row_params = {
                "vector": c["quant_params"],
                "named": c["named_values"],
                "candidate_sha256": c["candidate_sha256"],
                "evaluation_lineage": c["evaluation_lineage"],
                "scenario_candidate_ids": c["scenario_candidate_ids"],
                "scenario_fingerprints": c["scenario_fingerprints"],
            }
            batch_rows.append(
                EvaluationRow(
                    candidate_id=primary_scen_cand_id,
                    ordinal=step + c["cand_idx"],
                    coordinates=c["lineage"],
                    params=row_params,
                    pool_overrides=c["pool_overrides"],
                    metrics=row_metrics,
                    status="ok" if c["evaluation_ok"] else "failed",
                    economic_fingerprint=primary_fingerprint,
                    tags=(
                        f"algorithm:{algorithm}",
                        "lane:main",
                        f"ask_id:{c['ask_id']}",
                        f"bare_id:{c['cand_id']}",
                    ),
                )
            )

        if not batch_rows:
            raise RuntimeError("optimizer produced an empty completed batch")

        optimizer.tell(
            [c["quant_params"] for c in prepared_candidates],
            [c["loss"] for c in prepared_candidates],
            objectives=[c["objective"] for c in prepared_candidates],
            scores=[c["score_res"] for c in prepared_candidates],
        )

        rows.extend(batch_rows)
        if any(left.ordinal >= right.ordinal for left, right in zip(rows, rows[1:])):
            raise ValueError("evaluation table ordinals must be strictly increasing")
        step = len(rows)
        table = EvaluationTable(rows=rows, metadata={**immutable_table_metadata,
            "candidates_evaluated": step}, metric_projection=metric_projection)
        table_path = run_store.save_evaluation_table(actual_run_id, table)
        _save_checkpoint()
    eval_rows = rows
    if table is None:
        raise RuntimeError("optimizer produced no durable evaluation table")

    # 6. Extract Top-K & Winner SelectionRefs with exact replay candidate IDs
    sorted_rows = sorted(
        eval_rows,
        key=lambda r: r.metrics.get("loss", float("inf")),
    )
    if not sorted_rows:
        raise RuntimeError("TMRBCD produced no evaluated candidates")
    best_row = sorted_rows[0]

    top_k_refs: list[SelectionRef] = []
    for rank, r in enumerate(sorted_rows[:top_k_count]):
        top_k_refs.append(
            SelectionRef(
                run_id=actual_run_id,
                kind="optimizer_winner",
                index=r.ordinal,
                coordinate=r.coordinates,
                candidate_id=r.candidate_id,
                tags=(
                    f"rank:{rank}",
                    f"candidate:{r.candidate_id}",
                    f"policy_sha256:{verified_policy_sha256[:12]}",
                    f"algorithm:{algorithm}",
                ),
            )
        )

    winner_ref = top_k_refs[0] if top_k_refs else SelectionRef(
        run_id=actual_run_id,
        kind="optimizer_winner",
        index=best_row.ordinal,
        coordinate=best_row.coordinates,
        candidate_id=best_row.candidate_id,
        tags=(
            "rank:0",
            f"candidate:{best_row.candidate_id}",
            f"policy_sha256:{verified_policy_sha256[:12]}",
            f"algorithm:{algorithm}",
        ),
    )

    winner_payload = {
        "winner": winner_ref.to_dict(),
        "policy_id": opt_spec.policy_id,
        "policy_header": policy_header,
        "policy_source_sha256": verified_policy_sha256,
        "evaluator_identity": evaluator_ident_dict,
        "named_runtime": named_runtime_identity,
        "yb_settings": yb_settings,
        "pair": pair_spec.to_dict(),
        "scenario": primary_scenario.to_dict(),
        "scenarios": resolved_scenarios_info,
        "lineage": best_row.coordinates,
        "economic_fingerprint": best_row.economic_fingerprint,
        "scenario_fingerprints": best_row.params.get("scenario_fingerprints", {}),
        "scenario_candidate_ids": best_row.params.get("scenario_candidate_ids", {}),
        "params": best_row.params,
        "metrics": best_row.metrics,
    }
    winner_path = atomic_write_json(run_dir / "winner.json", winner_payload)

    topk_payload = {
        "top_k": [ref.to_dict() for ref in top_k_refs],
        "policy_id": opt_spec.policy_id,
        "policy_header": policy_header,
        "policy_source_sha256": verified_policy_sha256,
        "evaluator_identity": evaluator_ident_dict,
        "named_runtime": named_runtime_identity,
        "yb_settings": yb_settings,
        "pair": pair_spec.to_dict(),
        "scenario": primary_scenario.to_dict(),
        "scenarios": resolved_scenarios_info,
    }
    topk_path = atomic_write_json(run_dir / "topk.json", topk_payload)

    # Calculate SHA-256 and byte sizes of all written artifacts.
    table_sha = sha256_path(table_path)
    table_bytes = table_path.stat().st_size
    winner_sha = sha256_path(winner_path)
    winner_bytes = winner_path.stat().st_size
    topk_sha = sha256_path(topk_path)
    topk_bytes = topk_path.stat().st_size
    checkpoint_sha = sha256_path(checkpoint_file) if checkpoint_file.is_file() else None
    checkpoint_bytes = checkpoint_file.stat().st_size if checkpoint_file.is_file() else None

    artifacts_list = [
        {
            "path": "evaluation_table.npz",
            "kind": "evaluation_table",
            "sha256": table_sha,
            "bytes": table_bytes,
        },
        {
            "path": "winner.json",
            "kind": "winner",
            "sha256": winner_sha,
            "bytes": winner_bytes,
        },
        {
            "path": "topk.json",
            "kind": "topk",
            "sha256": topk_sha,
            "bytes": topk_bytes,
        },
    ]
    if checkpoint_sha:
        artifacts_list.append({
            "path": "checkpoint.json",
            "kind": "checkpoint",
            "sha256": checkpoint_sha,
            "bytes": checkpoint_bytes,
        })

    artifacts_list.extend({"path": path.relative_to(run_dir).as_posix(), "kind": kind,
        "sha256": sha256_path(path), "bytes": path.stat().st_size} for path, kind in (
            (selected_evaluator.artifact.receipt_path, "evaluator_artifact_receipt"),
            (selected_evaluator.binary_path, "evaluator_binary")))
    published_core = selected_evaluator.manifest_core(binary_override="evaluator_artifact/evaluator")

    # 7. Write Manifest with complete immutable replay identity & attested artifact hashes
    manifest_payload = new_optimization_manifest(
        run_id=actual_run_id,
        optimization_id=opt_spec.id,
        algorithm=algorithm,
        scenarios=opt_spec.scenarios,
        resolved_spec={
            **opt_spec.to_dict(),
            "pair": pair_spec.to_dict(),
            "scenario": primary_scenario.to_dict(),
            "scenario_specs": [s.to_dict() for s in scenarios],
            "policy": {
                "id": opt_spec.policy_id,
                "header_file": policy_header,
                "source_sha256": verified_policy_sha256,
                "policy_abi": policy_abi,
                "parameter_names": list(policy_parameter_names),
            },
            "resolved_scenarios": resolved_scenarios_info,
            "policy_header": policy_header,
            "policy_source_sha256": verified_policy_sha256,
            "yb_settings": yb_settings,
            "metric_projection": metric_projection.to_dict(),
            "execution_closure_sha256": execution_closure_sha256,
            "named_runtime": named_runtime_identity,
        },
        candidates_evaluated=len(eval_rows),
        best_candidate={
            "candidate_id": best_row.candidate_id,
            "params": best_row.params,
            "pool_overrides": best_row.pool_overrides,
            "metrics": best_row.metrics,
            "loss": best_row.metrics.get("loss"),
            "objective_value": best_row.metrics.get("objective_value"),
            "lineage": best_row.coordinates,
            "economic_fingerprint": best_row.economic_fingerprint,
            "scenario_candidate_ids": best_row.params.get("scenario_candidate_ids", {}),
            "scenario_fingerprints": best_row.params.get("scenario_fingerprints", {}),
        },
        core=published_core,
        artifacts=artifacts_list,
        table_ref={
            "path": "evaluation_table.npz",
            "row_count": len(eval_rows),
            "sha256": table_sha,
            "bytes": table_bytes,
        },
    )
    manifest_path = run_store.save_manifest(actual_run_id, manifest_payload)


    return OptimizationResult(
        run_id=actual_run_id,
        spec=opt_spec,
        table=table,
        winner=winner_ref,
        top_k=top_k_refs,
        manifest_path=manifest_path,
        table_path=table_path,
        winner_path=winner_path,
        topk_path=topk_path,
        best_score=dict(best_row.metrics),
        candidates_evaluated=len(eval_rows),
    )


def run_optimization(
    spec_or_path: str | os.PathLike[str] | OptimizationSpec,
    *,
    store: RunStore | None = None,
    run_id: str | None = None,
    resume: bool = False,
    budget: int | None = None,
    batch_size: int | None = None,
    top_k_count: int = 8,
    repository: Path | None = None,
    site: str = "local",
    blades: Sequence[str] = (),
    selected_evaluator: SelectedEvaluator,
) -> OptimizationResult:
    if not isinstance(selected_evaluator, SelectedEvaluator):
        raise TypeError("selected_evaluator must be a SelectedEvaluator")
    context = _execution_context(repository, store)
    root = context.project_root
    distributed = site != "local" or bool(blades)
    profile = load_site_profile(site, root=root) if distributed else None
    if profile is not None and profile.cluster.transport != "shared_nfs":
        raise ValueError("selected-evaluator optimization requires shared_nfs transport")
    args = dict(store=store, run_id=run_id, resume=resume, budget=budget, batch_size=batch_size,
                top_k_count=top_k_count, repository=repository, site=site, blades=blades,
                selected_evaluator=selected_evaluator)
    if profile is not None:
        spec = spec_or_path if isinstance(spec_or_path, OptimizationSpec) else load_optimization_spec(spec_or_path, repository=root)
        actual_run_id = run_id or spec.id or f"opt_{uuid.uuid4().hex[:12]}"
        with shared_run_lease(profile, actual_run_id) as lease:
            previous = os.environ.get("FXSIM_RUN_LEASE_TOKEN")
            os.environ["FXSIM_RUN_LEASE_TOKEN"] = lease.token
            try:
                args.update(run_id=actual_run_id, repository=root)
                return _run_optimization(spec, **args)
            finally:
                if previous is None:
                    os.environ.pop("FXSIM_RUN_LEASE_TOKEN", None)
                else:
                    os.environ["FXSIM_RUN_LEASE_TOKEN"] = previous
    return _run_optimization(spec_or_path, **args)


def status_optimization(
    run_id_or_path: str | os.PathLike[str],
    *,
    store: RunStore | None = None,
    repository: Path | None = None,
) -> OptimizationStatus:
    """Query current status, candidate count, and best objective of an optimization run."""
    context = _execution_context(repository, store)
    run_store = store if store is not None else RunStore(context)

    manifest = run_store.load_manifest(run_id_or_path, expected_kind="optimization")
    run_id = manifest["run_id"]
    run_dir = run_store.get_run_dir(run_id)

    opt_info = manifest.get("optimization", {})
    best_cand = opt_info.get("best_candidate", {})

    status_str = "completed" if (run_dir / "winner.json").is_file() else "running"

    return OptimizationStatus(
        run_id=run_id,
        status=status_str,
        candidates_evaluated=int(opt_info.get("candidates_evaluated", 0)),
        best_loss=float(best_cand.get("loss", float("inf"))),
        best_objective=float(best_cand.get("objective_value", float("-inf"))),
        best_params=list(best_cand.get("params", {}).get("vector", [])),
        elapsed_seconds=float(manifest.get("elapsed_seconds", 0.0)),
        best_score=dict(best_cand.get("metrics", {})),
        updated_at=str(manifest.get("updated_at", "")),
    )


def collect_optimization(
    run_id_or_path: str | os.PathLike[str],
    *,
    store: RunStore | None = None,
    repository: Path | None = None,
) -> OptimizationResult:
    """Collect finalized artifacts from an optimization run directory."""
    context = _execution_context(repository, store)
    root = context.project_root
    run_store = store if store is not None else RunStore(context)

    manifest = run_store.load_manifest(run_id_or_path, expected_kind="optimization")
    run_id = manifest["run_id"]
    run_dir = run_store.get_run_dir(run_id)

    table, table_path = load_attested_evaluation_table(
        manifest,
        run_dir=run_dir,
    )

    winner_path = find_attested_artifact(
        manifest,
        run_dir=run_dir,
        kind="winner",
    )
    with winner_path.open("r", encoding="utf-8") as f:
        winner_payload = json.load(f)
    winner_dict = winner_payload.get("winner", winner_payload)
    winner_ref = SelectionRef.from_dict(winner_dict)

    top_k_refs: list[SelectionRef] = []
    topk_path = find_attested_artifact(
        manifest,
        run_dir=run_dir,
        kind="topk",
    )
    with topk_path.open("r", encoding="utf-8") as f:
        topk_payload = json.load(f)
        topk_list = topk_payload.get(
            "top_k",
            topk_payload if isinstance(topk_payload, list) else [],
        )
        top_k_refs = [SelectionRef.from_dict(item) for item in topk_list]

    resolved_spec = manifest.get("resolved_spec", {})
    if not isinstance(resolved_spec, Mapping):
        raise ValueError("optimization manifest resolved_spec must be an object")
    opt_info = manifest.get("optimization", {})
    scenario_ids = opt_info.get("scenarios", ()) if isinstance(opt_info, Mapping) else ()

    def _mapping_field(name: str) -> dict[str, Any]:
        value = resolved_spec.get(name, {})
        if not isinstance(value, Mapping):
            raise ValueError(f"optimization manifest resolved_spec.{name} must be an object")
        return dict(value)

    source_path_value = resolved_spec.get("source_path")
    spec = OptimizationSpec(
        id=resolved_spec.get("id", run_id),
        pair_id=resolved_spec.get("pair_id", ""),
        policy_id=resolved_spec.get("policy_id", ""),
        algorithm=resolved_spec.get("algorithm", "tmrbcd"),
        scenarios=tuple(str(value) for value in scenario_ids),
        parameter_space=_mapping_field("parameter_space"),
        optimizer_config=_mapping_field("optimizer_config"),
        scoring_config=_mapping_field("scoring_config"),
        tags=tuple(str(value) for value in resolved_spec.get("tags", ())),
        source_path=Path(str(source_path_value)) if source_path_value else None,
    )

    best_cand = opt_info.get("best_candidate", {})

    return OptimizationResult(
        run_id=run_id,
        spec=spec,
        table=table,
        winner=winner_ref,
        top_k=top_k_refs,
        manifest_path=run_dir / "manifest.json",
        table_path=table_path,
        winner_path=winner_path,
        topk_path=topk_path,
        best_score=dict(best_cand.get("metrics", {})),
        candidates_evaluated=int(opt_info.get("candidates_evaluated", len(table.rows))),
    )


__all__ = [
    "OptimizationStatus",
    "OptimizationResult",
    "run_optimization",
    "status_optimization",
    "collect_optimization",
]
