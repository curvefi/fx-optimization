"""Complete local and distributed optimization execution runtime with full immutable replay inputs & lineage."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import uuid
from concurrent.futures import ThreadPoolExecutor
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
from ..evaluation.client import HarnessClient, ScenarioHarnessClient, SubprocessHarnessClient
from ..evaluation.identity import (
    VerifiedEvaluator,
    validate_evaluator_identity,
    verified_evaluator_from_payload,
)
from curve_fx_harness_client.models import CandidateSpec, ObservationSpec
from ..evaluation.selection import SelectionRef
from ..execution.adapter import SSHProcessAdapter
from ..execution.site import SiteProfile, load_site_profile
from ..execution.staging import remote_run_paths, scoped_remote_path
from ..specs.common import repository_root
from ..specs.optimization import OptimizationSpec, load_optimization_spec
from ..specs.pair import load_pair_spec
from ..specs.policy import load_policy_spec
from ..specs.scenario import load_scenario_spec
from .profiles import create_lattice_spec, profile_from_policy_spec, quantized
from .requests import split_request
from .nevergrad_adapter import NevergradTwoPointsDEOptimizer
from .worker import OptimizationBundleResult, create_work_bundle
from .scoring import (
    SCORE_FX_LP_E15_SLIPPAGE_V1_KEY,
    SCORE_VERSION,
    loss_from_score,
    objective_failure_count,
    normalize_score_key,
    score_objective_value,
    score_scenarios,
)
from .tmrbcd import TmrbcdOptimizer

OPTIMIZATION_CHECKPOINT_SCHEMA_VERSION = 5
EVALUATION_JOURNAL_FILENAME = "evaluation_journal.jsonl"


def _read_evaluation_journal(journal_path: Path) -> list[EvaluationRow]:
    """Read the plain JSONL evaluation journal; malformed rows fail naturally."""
    if not journal_path.is_file():
        return []
    rows: list[EvaluationRow] = []
    with journal_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                raw_row = json.loads(line)
                row = EvaluationRow.from_dict(raw_row)
            except (
                UnicodeDecodeError,
                json.JSONDecodeError,
                TypeError,
                ValueError,
                KeyError,
                AttributeError,
            ) as exc:
                raise ValueError(
                    f"Resume evaluation journal has a malformed row at line {line_number}"
                ) from exc
            rows.append(row)
    return rows


def _dispatch_remote_bundle(
    bundle: Any,
    *,
    run_dir: Path,
    site: SiteProfile,
    blade: str,
    expected_ask_ids: frozenset[str],
) -> OptimizationBundleResult:
    """Dispatch one immutable optimization bundle, reusing an attested local result."""
    bundle_dir = run_dir / "bundles"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    local_bundle = bundle_dir / f"{bundle.bundle_id}.json"
    local_result = bundle_dir / f"{bundle.bundle_id}.result.json"
    bundle.to_json(local_bundle)

    def load_checked() -> OptimizationBundleResult:
        result = OptimizationBundleResult.from_json(local_result)
        if (result.bundle_id, result.bundle_sha256, result.island_id) != (
            bundle.bundle_id,
            bundle.bundle_sha256,
            bundle.island_id,
        ):
            raise ValueError(f"remote result does not attest bundle {bundle.bundle_id}")
        ask_ids = [str(item["ask_id"]) for item in result.results]
        if len(ask_ids) != len(set(ask_ids)) or frozenset(ask_ids) != expected_ask_ids:
            raise ValueError(f"remote result has non-exact ask_id coverage for {bundle.bundle_id}")
        return result

    if local_result.is_file():
        return load_checked()
    paths = remote_run_paths(bundle.run_id, remote_base=site.cluster.remote_base)
    remote_bundle = scoped_remote_path(bundle.run_id, f"inputs/{local_bundle.name}", remote_base=site.cluster.remote_base)
    remote_result = scoped_remote_path(bundle.run_id, f"results/{local_result.name}", remote_base=site.cluster.remote_base)
    ssh = SSHProcessAdapter(ssh_config=site.ssh)
    setup = ssh.run_ssh(blade, f"mkdir -p {shlex.quote(str(paths['inputs']))} {shlex.quote(str(paths['results']))}")
    if not setup.ok:
        raise RuntimeError(f"failed to prepare remote run on {blade}: {setup.stderr}")
    uploaded = ssh.rsync_upload(local_bundle, blade, str(remote_bundle))
    if not uploaded.ok:
        raise RuntimeError(f"failed to stage optimization bundle {bundle.bundle_id}: {uploaded.stderr}")
    worker = shlex.quote(site.cluster.worker_command)
    repository = shlex.quote(str(site.cluster.repository_root))
    harness = shlex.quote(str(site.harness.remote_binary_path or site.harness.binary_name))
    command = (
        f"cd {repository} && "
        f"{worker} optimize worker {shlex.quote(str(remote_bundle))} "
        f"--root {repository} "
        f"--harness {harness} --out {shlex.quote(str(remote_result))}"
    )
    executed = ssh.run_ssh(blade, command)
    if not executed.ok:
        raise RuntimeError(f"remote optimization worker failed on {blade}: {executed.stderr}")
    downloaded = ssh.rsync_download(blade, str(remote_result), local_result)
    if not downloaded.ok:
        raise RuntimeError(f"failed to collect optimization result {bundle.bundle_id}: {downloaded.stderr}")
    return load_checked()


def _inspect_remote_core(site: SiteProfile, blade: str) -> dict[str, Any]:
    """Read the deployed evaluator through the canonical hello schema."""
    binary = str(site.harness.remote_binary_path or site.harness.binary_name)
    result = SSHProcessAdapter(ssh_config=site.ssh).run_ssh(
        blade,
        f"{shlex.quote(binary)} --identity-json",
    )
    if not result.ok:
        raise RuntimeError(f"failed to inspect remote evaluator on {blade}: {result.stderr}")
    try:
        payload = json.loads(result.stdout)
        if not isinstance(payload, Mapping):
            raise TypeError("identity payload is not an object")
        evaluator = verified_evaluator_from_payload(payload, path=binary)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"remote evaluator on {blade} returned invalid identity JSON") from exc
    core = evaluator.to_core_dict(binary_override=binary)
    core["site"] = site.name
    core["remote_sha256"] = evaluator.sha256
    return core


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


def _resolve_evaluator_client(
    client: HarnessClient | None,
    root: Path,
    *,
    scenarios: Sequence[Any] = (),
) -> HarnessClient:
    if client is not None:
        if scenarios and isinstance(client, SubprocessHarnessClient):
            used_original = False

            def factory(_scenario: Any) -> HarnessClient:
                nonlocal used_original
                if not used_original:
                    used_original = True
                    return client
                return client.clone()

            return ScenarioHarnessClient(scenarios, factory)
        return client

    search_paths = [
        root / "bin" / "arb_evaluator_ld",
        root.parent / "curve-fx-arb-harness" / "build" / "arb_evaluator_ld",
        root.parent / "curve-fx-arb-harness" / "bin" / "arb_evaluator_ld",
        Path("/usr/local/bin/arb_evaluator_ld"),
    ]
    for p in search_paths:
        if p.is_file():
            if scenarios:
                return ScenarioHarnessClient(
                    scenarios,
                    lambda _scenario, binary=p: SubprocessHarnessClient(binary, repository=root),
                )
            return SubprocessHarnessClient(p, repository=root)

    raise ValueError(
        "No active HarnessClient provided and no local evaluator binary found. "
        "Explicitly pass a HarnessClient or ensure arb_evaluator_ld is built."
    )


def _run_optimization(
    spec_or_path: str | os.PathLike[str] | OptimizationSpec,
    *,
    store: RunStore | None = None,
    client: HarnessClient | None = None,
    run_id: str | None = None,
    resume: bool = False,
    budget: int | None = None,
    batch_size: int | None = None,
    top_k_count: int = 8,
    repository: Path | None = None,
    site: str = "local",
    blades: Sequence[str] = (),
    _owned_clients: list[ScenarioHarnessClient],
) -> OptimizationResult:
    """Execute an optimization run against the harness evaluator."""
    root = repository.resolve() if repository is not None else repository_root()
    run_store = store if store is not None else RunStore(root)
    distributed = site != "local" or bool(blades)
    if distributed and client is not None:
        raise ValueError("an explicit local HarnessClient cannot be used with distributed optimization")
    site_profile = load_site_profile(site, root=root) if distributed else None
    target_blades = (
        list(blades) or list(site_profile.cluster.blades)
        if site_profile is not None
        else []
    )
    if distributed and not target_blades:
        raise ValueError(f"site {site_profile.name!r} has no blades configured")

    if isinstance(spec_or_path, OptimizationSpec):
        opt_spec = spec_or_path
    else:
        opt_spec = load_optimization_spec(spec_or_path, repository=root)

    actual_run_id = run_id or opt_spec.id or f"opt_{uuid.uuid4().hex[:12]}"

    # 1. Resolve the sole PolicySpec authority and evaluator identity.
    policy_spec = load_policy_spec(opt_spec.policy_id, repository=root)
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
    profile = profile_from_policy_spec(
        policy_spec, opt_spec.parameter_space, template_json=template_json
    )
    eval_client = None if distributed else _resolve_evaluator_client(
        client,
        root,
        scenarios=scenarios,
    )
    if isinstance(eval_client, ScenarioHarnessClient):
        _owned_clients.append(eval_client)
    if distributed:
        assert site_profile is not None
        remote_identities = [_inspect_remote_core(site_profile, blade) for blade in target_blades]
        evaluator_ident_dict = remote_identities[0]
        for blade, identity in zip(target_blades, remote_identities, strict=True):
            for key, expected in (
                ("policy_id", profile.name),
                ("policy_source_sha256", profile.source_sha256),
                ("policy_abi", profile.policy_abi),
                ("policy_parameter_count", profile.n_params()),
            ):
                if str(identity.get(key, "")).lower() != str(expected).lower():
                    raise ValueError(
                        f"remote evaluator on {blade} reports {key}={identity.get(key)!r}, expected {expected!r}"
                    )
            if identity["sha256"] != evaluator_ident_dict["sha256"]:
                raise ValueError("all selected blades must run the same evaluator binary SHA-256")
    else:
        assert eval_client is not None
        evaluator_ident = eval_client.prepare()
        if not isinstance(evaluator_ident, VerifiedEvaluator):
            raise TypeError("HarnessClient.prepare() must return a VerifiedEvaluator")
        validate_evaluator_identity(
            evaluator_ident,
            expected_policy_id=profile.name,
            expected_policy_source_sha256=profile.source_sha256,
            expected_policy_abi=profile.policy_abi,
            expected_policy_parameter_count=profile.n_params(),
        )
        evaluator_ident_dict = evaluator_ident.to_dict()
    evaluator_metric_fields = tuple(str(field) for field in evaluator_ident_dict.get("metric_fields", ()))
    if not evaluator_metric_fields:
        raise ValueError("evaluator identity must attest non-empty metric_fields")
    metric_projection = MetricProjection.from_fields(
        evaluator_metric_fields,
        projection_id="optimization-primary-scenario",
    )

    # Policy source must exist and match the checked-in attestation.
    policy_header_path = (root / profile.header_file).resolve()
    if not policy_header_path.is_file():
        raise FileNotFoundError(f"compiled policy header not found: {policy_header_path}")
    calculated_sha = sha256_path(policy_header_path)
    if calculated_sha != profile.source_sha256:
        raise ValueError(
            f"Policy closure hash mismatch for {profile.header_file}: "
            f"expected {profile.source_sha256}, calculated {calculated_sha}"
        )
    verified_policy_sha256 = calculated_sha

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
    lattice = create_lattice_spec(profile)
    seed = int(opt_spec.optimizer_config.get("seed", 42))
    if algorithm == "tmrbcd":
        optimizer = TmrbcdOptimizer(
            lattice=lattice,
            initial_params=profile.initial_seed,
            budget=total_budget,
            seed=seed,
        )
    elif algorithm == "nevergrad_two_points_de":
        optimizer = NevergradTwoPointsDEOptimizer(
            lattice=lattice,
            initial_params=profile.initial_seed,
            budget=total_budget,
            seed=seed,
            num_workers=eval_batch_size,
        )
    else:
        raise ValueError(f"unsupported optimization algorithm {algorithm!r}")

    run_identity = {
        "optimization": opt_spec.to_dict(),
        "policy": policy_spec.to_dict(),
        "profile": profile.to_dict(),
        "pair": pair_spec.to_dict(),
        "scenarios": resolved_scenarios_info,
        "evaluator": evaluator_ident_dict,
        "score_key": score_key,
    }
    run_identity_sha256 = hashlib.sha256(canonical_json_bytes(run_identity)).hexdigest()
    run_dir = (
        run_store.get_run_dir(actual_run_id)
        if resume
        else run_store.allocate_run_dir("optimization", actual_run_id)
    )

    # 3. Restore the optimizer and the exact committed prefix of the durable row journal.
    checkpoint_file = run_dir / "checkpoint.json"
    journal_file = run_dir / EVALUATION_JOURNAL_FILENAME
    rows: list[EvaluationRow] = []
    replayed_tail = False

    if resume:
        if not checkpoint_file.is_file():
            raise ValueError("Resume checkpoint is missing; start a new immutable run")
        with checkpoint_file.open("r", encoding="utf-8") as f:
            cp_data = json.load(f)
        if cp_data.get("schema_version") != OPTIMIZATION_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("Resume checkpoint schema is stale; start a new immutable run")

        # Validate the immutable run identity; cheap and catches real mis-resume.
        cp_policy_id = cp_data.get("policy_id")
        if cp_policy_id and cp_policy_id != opt_spec.policy_id:
            raise ValueError(
                f"Resume identity mismatch: checkpoint policy_id {cp_policy_id!r} != spec {opt_spec.policy_id!r}"
            )
        cp_score_key = cp_data.get("score_key")
        if cp_score_key and cp_score_key != score_key:
            raise ValueError(
                f"Resume identity mismatch: checkpoint score_key {cp_score_key!r} != spec {score_key!r}"
            )
        cp_pair_id = cp_data.get("pair_id")
        if cp_pair_id and cp_pair_id != opt_spec.pair_id:
            raise ValueError(
                f"Resume identity mismatch: checkpoint pair_id {cp_pair_id!r} != spec {opt_spec.pair_id!r}"
            )
        if cp_data.get("run_identity_sha256") != run_identity_sha256:
            raise ValueError("Resume identity mismatch: policy, lattice, scenario, or evaluator changed")

        # The journal is the plain row log: rows already in it are already
        # evaluated and are skipped, not re-run.  A crash between a journal
        # append and the checkpoint save can leave the journal one batch ahead
        # of the restored optimizer; re-drive the optimizer over that journaled
        # tail so ask/tell stay aligned without re-evaluation.
        rows = _read_evaluation_journal(journal_file)
        opt_state = cp_data.get("optimizer_state")
        if not isinstance(opt_state, Mapping):
            raise ValueError("Resume checkpoint optimizer_state must be an object")
        optimizer.restore(opt_state)
        if optimizer.step > len(rows):
            raise ValueError(
                "Resume checkpoint is ahead of the evaluation journal; start a new immutable run"
            )
        for start in range(optimizer.step, len(rows), eval_batch_size):
            chunk = rows[start : start + eval_batch_size]
            optimizer.ask(len(chunk))
            optimizer.tell(
                [row.params["vector"] for row in chunk],
                [float(row.metrics["loss"]) for row in chunk],
                objectives=[float(row.metrics["objective_value"]) for row in chunk],
                scores=[dict(row.metrics) for row in chunk],
            )
            replayed_tail = True

    if (
        resume
        and len(rows) >= total_budget
        and all((run_dir / name).is_file() for name in ("manifest.json", "evaluation_table.npz", "winner.json", "topk.json"))
    ):
        return collect_optimization(
            actual_run_id,
            store=run_store,
            repository=root,
        )

    step = len(rows)
    # Common evaluation table metadata header
    immutable_table_metadata = {
        "run_id": actual_run_id,
        "optimization_id": opt_spec.id,
        "algorithm": algorithm,
        "policy_id": opt_spec.policy_id,
        "policy_header": profile.header_file,
        "policy_source_sha256": verified_policy_sha256,
        "policy_abi": profile.policy_abi,
        "parameter_names": list(profile.parameter_names),
        "lattice": profile.to_dict(),
        "pair_id": opt_spec.pair_id,
        "pair": pair_spec.to_dict(),
        "scenario": primary_scenario.to_dict(),
        "scenarios": resolved_scenarios_info,
        "evaluator_identity": evaluator_ident_dict,
        "yb_settings": yb_settings,
        "score_key": score_key,
    }

    # Publish a compact checkpoint after each journal batch is appended.
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
            "step": step,
            "optimizer_state": opt_snapshot,
        }
        atomic_write_json(checkpoint_file, cp_payload)

    if replayed_tail:
        # A journaled tail can exhaust the budget without the loop running;
        # persist the caught-up optimizer state before final publication.
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
        for cand_idx, raw_params in enumerate(raw_asks):
            quant_params = quantized(profile, raw_params)
            policy_params, pool_overrides = split_request(profile, quant_params)
            cand_id = f"c_{step + cand_idx:05d}"
            ask_id = f"ask_{step + cand_idx:06d}"
            prepared_candidates.append(
                {
                    "cand_id": cand_id,
                    "cand_idx": cand_idx,
                    "ask_id": ask_id,
                    "quant_params": quant_params,
                    "pool_overrides": pool_overrides,
                    "policy_params": policy_params,
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

        if distributed:
            assert site_profile is not None
            assert target_blades
            grouped: dict[str, list[dict[str, Any]]] = {
                blade: [] for blade in target_blades
            }
            for index, candidate in enumerate(prepared_candidates):
                grouped[target_blades[index % len(target_blades)]].append(candidate)
            jobs: list[tuple[str, list[dict[str, Any]], Any, str, frozenset[str]]] = []
            for blade, blade_candidates in grouped.items():
                if not blade_candidates:
                    continue
                bundle = create_work_bundle(
                    run_id=actual_run_id,
                    optimization_id=opt_spec.id,
                    island_id=blade,
                    step=step,
                    profile=profile,
                    pair_spec=pair_spec,
                    scenarios=scenarios,
                    evaluator_identity=evaluator_ident_dict,
                    yb_settings=yb_settings,
                    score_key=score_key,
                    proposals=[
                        {
                            "ordinal": candidate["cand_idx"],
                            "ask_id": candidate["ask_id"],
                            "params": candidate["quant_params"],
                            "pool_overrides": candidate["pool_overrides"],
                        }
                        for candidate in blade_candidates
                    ],
                    pool_overrides={},
                )
                jobs.append(
                    (
                        blade,
                        blade_candidates,
                        bundle,
                        blade,
                        frozenset(str(candidate["ask_id"]) for candidate in blade_candidates),
                    )
                )

            def dispatch(job: tuple[str, list[dict[str, Any]], Any, str, frozenset[str]]) -> OptimizationBundleResult:
                _, _, bundle, blade, ask_ids = job
                return _dispatch_remote_bundle(
                    bundle,
                    run_dir=run_dir,
                    site=site_profile,
                    blade=blade,
                    expected_ask_ids=ask_ids,
                )

            with ThreadPoolExecutor(max_workers=min(len(target_blades), len(jobs))) as executor:
                remote_results = list(executor.map(dispatch, jobs))
            for (_, island_candidates, _, _, _), remote_result in zip(jobs, remote_results, strict=True):
                by_ask_id = {str(item["ask_id"]): item for item in remote_result.results}
                for candidate in island_candidates:
                    result = by_ask_id.get(str(candidate["ask_id"]))
                    if result is None:
                        raise ValueError(f"remote bundle omitted candidate ask_id {candidate['ask_id']}")
                    candidate["loss"] = float(result["loss"])
                    candidate["objective"] = float(result["objective"])
                    candidate["score_res"] = dict(result["score_res"])
                    candidate["evaluation_ok"] = result.get("status") == "ok"
                    candidate["primary_metrics"] = dict(result.get("primary_metrics", {}))
                    candidate["scenario_candidate_ids"] = dict(result.get("scenario_candidate_ids", {}))
                    candidate["scenario_fingerprints"] = dict(result.get("scenario_fingerprints", {}))
        else:
            # Evaluate across scenarios in batched calls.
            assert eval_client is not None
            for scen in scenarios:
                eval_client.open_session(scen)
                batch_requests = []
                for c in prepared_candidates:
                    scen_cand_id = f"{c['cand_id']}_{scen.id}"
                    c["scenario_candidate_ids"][scen.id] = scen_cand_id
                    batch_requests.append(
                        CandidateSpec(
                            ordinal=c["cand_idx"],
                            candidate_id=scen_cand_id,
                            policy_params=c["policy_params"],
                            pool_overrides=c["pool_overrides"],
                        )
                    )

                batch_resp = eval_client.evaluate_batch(
                    batch_requests,
                    observation=ObservationSpec(kind="summary"),
                )
                if batch_resp.status != "complete":
                    raise RuntimeError(
                        f"evaluator returned incomplete batch for scenario {scen.id!r}: "
                        f"{batch_resp.status!r}"
                    )
                results_by_ordinal = {r.ordinal: r for r in batch_resp.results}
                for c in prepared_candidates:
                    res = results_by_ordinal.get(c["cand_idx"])
                    if res is not None:
                        c["evaluation_ok"] = c["evaluation_ok"] and res.status == "ok"
                        res_metrics = dict(res.metrics)
                        res_metrics["ok"] = res.status == "ok"
                        c["scenario_results"].append(res_metrics)
                        if scen.id == primary_scenario.id:
                            c["primary_metrics"] = dict(res.metrics)
                        if res.economic_fingerprint:
                            c["scenario_fingerprints"][scen.id] = res.economic_fingerprint
                    else:
                        c["evaluation_ok"] = False
                        c["scenario_results"].append({"ok": False})

            # Score candidates and build evaluation rows.
            for c in prepared_candidates:
                score_res = score_scenarios(c["scenario_results"], require_yb=require_yb)
                obj_val = score_objective_value(score_res, score_key)
                score_res["objective_value"] = obj_val
                score_res["objective_failures"] = objective_failure_count(score_res, score_key)
                c["loss"] = loss_from_score(score_res)
                c["objective"] = obj_val
                c["score_res"] = score_res

        # Build canonical evaluation rows from either local scoring or attested worker results.
        batch_rows: list[EvaluationRow] = []
        for c in prepared_candidates:
            score_res = c["score_res"]
            loss_val = c["loss"]
            obj_val = c["objective"]
            primary_fingerprint = c["scenario_fingerprints"].get(primary_scenario.id)
            primary_scen_cand_id = c["scenario_candidate_ids"].get(primary_scenario.id, c["cand_id"])
            row_metrics = {
                **c["primary_metrics"],
                "loss": loss_val,
                "objective_value": obj_val,
                "scenario_candidate_ids": c["scenario_candidate_ids"],
                "scenario_fingerprints": c["scenario_fingerprints"],
                **score_res,
            }
            batch_rows.append(
                EvaluationRow(
                    candidate_id=primary_scen_cand_id,
                    ordinal=step + c["cand_idx"],
                    coordinates=c["lineage"],
                    params={
                        "vector": c["quant_params"],
                        "named": dict(
                            zip(
                                (
                                    *profile.parameter_names,
                                    *(dim.name for dim in profile.pool_dims),
                                ),
                                c["quant_params"],
                                strict=True,
                            )
                        ),
                    },
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

        # Append the batch to the plain JSONL row log, then checkpoint.
        # Durability invariant: flush the journal stream (user-space buffer to
        # OS) before _save_checkpoint(), so the durable JSONL on disk never lags
        # the checkpoint's step/candidates_evaluated.  fsync is intentionally
        # omitted: a genuine OS-crash partial write still fails loudly at resume
        # via the line-numbered malformed-row error in _read_evaluation_journal.
        rows.extend(batch_rows)
        with journal_file.open("ab") as stream:
            stream.write(
                b"".join(canonical_json_bytes(row.to_dict()) + b"\n" for row in batch_rows)
            )
            stream.flush()
        step += len(batch_rows)
        _save_checkpoint()
    eval_rows = rows
    # 5. Build final EvaluationTable with complete immutable metadata & sort canonically
    final_metadata = {
        **immutable_table_metadata,
        "candidates_evaluated": len(eval_rows),
    }

    table = EvaluationTable(
        rows=eval_rows,
        metadata=final_metadata,
        metric_projection=metric_projection,
    )
    table.sort_canonical()
    table_path = run_store.save_evaluation_table(actual_run_id, table)

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
        "policy_header": profile.header_file,
        "policy_source_sha256": verified_policy_sha256,
        "evaluator_identity": evaluator_ident_dict,
        "yb_settings": yb_settings,
        "pair": pair_spec.to_dict(),
        "scenario": primary_scenario.to_dict(),
        "scenarios": resolved_scenarios_info,
        "lineage": best_row.coordinates,
        "economic_fingerprint": best_row.economic_fingerprint,
        "scenario_fingerprints": best_row.metrics.get("scenario_fingerprints", {}),
        "scenario_candidate_ids": best_row.metrics.get("scenario_candidate_ids", {}),
        "metrics": best_row.metrics,
    }
    winner_path = atomic_write_json(run_dir / "winner.json", winner_payload)

    topk_payload = {
        "top_k": [ref.to_dict() for ref in top_k_refs],
        "policy_id": opt_spec.policy_id,
        "policy_header": profile.header_file,
        "policy_source_sha256": verified_policy_sha256,
        "evaluator_identity": evaluator_ident_dict,
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
    journal_sha = sha256_path(journal_file)
    journal_artifact_bytes = journal_file.stat().st_size
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
            "path": "evaluation_journal.jsonl",
            "kind": "evaluation_journal",
            "sha256": journal_sha,
            "bytes": journal_artifact_bytes,
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
                "header_file": profile.header_file,
                "source_sha256": verified_policy_sha256,
                "policy_abi": profile.policy_abi,
                "parameter_names": list(profile.parameter_names),
            },
            "resolved_scenarios": resolved_scenarios_info,
            "policy_header": profile.header_file,
            "policy_source_sha256": verified_policy_sha256,
            "yb_settings": yb_settings,
            "metric_projection": metric_projection.to_dict(),
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
            "scenario_candidate_ids": best_row.metrics.get("scenario_candidate_ids", {}),
            "scenario_fingerprints": best_row.metrics.get("scenario_fingerprints", {}),
        },
        core=evaluator_ident.to_core_dict() if not distributed else evaluator_ident_dict,
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
    client: HarnessClient | None = None,
    run_id: str | None = None,
    resume: bool = False,
    budget: int | None = None,
    batch_size: int | None = None,
    top_k_count: int = 8,
    repository: Path | None = None,
    site: str = "local",
    blades: Sequence[str] = (),
) -> OptimizationResult:
    """Execute an optimization run and close every internally pooled evaluator."""
    owned_clients: list[ScenarioHarnessClient] = []
    try:
        return _run_optimization(
            spec_or_path,
            store=store,
            client=client,
            run_id=run_id,
            resume=resume,
            budget=budget,
            batch_size=batch_size,
            top_k_count=top_k_count,
            repository=repository,
            site=site,
            blades=blades,
            _owned_clients=owned_clients,
        )
    finally:
        for owned_client in owned_clients:
            owned_client.close()


def status_optimization(
    run_id_or_path: str | os.PathLike[str],
    *,
    store: RunStore | None = None,
    repository: Path | None = None,
) -> OptimizationStatus:
    """Query current status, candidate count, and best objective of an optimization run."""
    root = repository.resolve() if repository is not None else repository_root()
    run_store = store if store is not None else RunStore(root)

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
    root = repository.resolve() if repository is not None else repository_root()
    run_store = store if store is not None else RunStore(root)

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
