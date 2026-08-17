"""Immutable TMRBCD work bundles and worker execution for blade transport."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..artifacts.io import atomic_write_json, canonical_json_bytes
from ..evaluation.client import HarnessClient, ScenarioHarnessClient, SubprocessHarnessClient
from ..evaluation.identity import VerifiedEvaluator, validate_evaluator_identity
from curve_fx_harness_client.models import CandidateSpec, ObservationSpec
from ..specs.common import canonical_primitive, serializable
from ..specs.pair import PairSpec
from ..specs.scenario import ScenarioSpec
from .profiles import Profile, quantized
from .requests import split_request
from .scoring import (
    loss_from_score,
    objective_failure_count,
    normalize_score_key,
    score_objective_value,
    score_scenarios,
)

WORK_BUNDLE_SCHEMA_VERSION = "fxsim_opt_work_bundle_v2"
BUNDLE_RESULT_SCHEMA_VERSION = "fxsim_opt_bundle_result_v3"


@dataclass(frozen=True)
class OptimizationWorkBundle:
    """An immutable, content-hashed unit of optimization work dispatched to a distributed worker."""

    bundle_id: str
    run_id: str
    optimization_id: str
    island_id: str
    step: int
    policy_id: str
    policy_header: str
    policy_source_sha256: str
    policy_abi: str
    policy_parameter_count: int
    pool_overrides: dict[str, Any]
    pair_spec: dict[str, Any]
    scenarios: tuple[dict[str, Any], ...]
    evaluator_identity: dict[str, Any]
    yb_settings: dict[str, Any]
    score_key: str
    proposals: tuple[dict[str, Any], ...]
    bundle_version: str = WORK_BUNDLE_SCHEMA_VERSION
    bundle_sha256: str = ""

    def __post_init__(self) -> None:
        if not self.bundle_sha256:
            digest = hashlib.sha256(canonical_json_bytes(self._hashable_dict())).hexdigest()
            object.__setattr__(self, "bundle_sha256", digest)

    def _hashable_dict(self) -> dict[str, Any]:
        return {
            "bundle_version": self.bundle_version,
            "bundle_id": self.bundle_id,
            "run_id": self.run_id,
            "optimization_id": self.optimization_id,
            "island_id": self.island_id,
            "step": self.step,
            "policy_id": self.policy_id,
            "policy_header": self.policy_header,
            "policy_source_sha256": self.policy_source_sha256,
            "policy_abi": self.policy_abi,
            "policy_parameter_count": self.policy_parameter_count,
            "pool_overrides": serializable(self.pool_overrides),
            "pair_spec": serializable(self.pair_spec),
            "scenarios": [serializable(s) for s in self.scenarios],
            "evaluator_identity": serializable(self.evaluator_identity),
            "yb_settings": serializable(self.yb_settings),
            "score_key": self.score_key,
            "proposals": [serializable(p) for p in self.proposals],
        }

    def to_dict(self) -> dict[str, Any]:
        data = self._hashable_dict()
        data["bundle_sha256"] = self.bundle_sha256
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OptimizationWorkBundle:
        supplied_hash = str(data.get("bundle_sha256", ""))
        if not supplied_hash:
            raise ValueError("work bundle is missing bundle_sha256 attestation")
        bundle = cls(
            bundle_id=str(data["bundle_id"]),
            run_id=str(data["run_id"]),
            optimization_id=str(data["optimization_id"]),
            island_id=str(data["island_id"]),
            step=int(data["step"]),
            policy_id=str(data["policy_id"]),
            policy_header=str(data.get("policy_header", "")),
            policy_source_sha256=str(data.get("policy_source_sha256", "")),
            policy_abi=str(data.get("policy_abi", "twocrypto_policy_v1")),
            policy_parameter_count=int(data["policy_parameter_count"]),
            pool_overrides=dict(data.get("pool_overrides", {})),
            pair_spec=dict(data.get("pair_spec", {})),
            scenarios=tuple(dict(s) for s in data.get("scenarios", ())),
            evaluator_identity=dict(data.get("evaluator_identity", {})),
            yb_settings=dict(data.get("yb_settings", {})),
            score_key=str(
                data.get("score_key", "score_fx_lp_e15_slippage_v1")
            ),
            proposals=tuple(dict(p) for p in data.get("proposals", ())),
            bundle_version=str(data.get("bundle_version", WORK_BUNDLE_SCHEMA_VERSION)),
            bundle_sha256=supplied_hash,
        )
        expected_hash = hashlib.sha256(canonical_json_bytes(bundle._hashable_dict())).hexdigest()
        if supplied_hash != expected_hash:
            raise ValueError(f"work bundle {bundle.bundle_id} hash mismatch")
        return bundle

    def to_json(self, path: Path | str) -> Path:
        return atomic_write_json(path, self.to_dict())

    @classmethod
    def from_json(cls, path: Path | str) -> OptimizationWorkBundle:
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


@dataclass(frozen=True)
class OptimizationBundleResult:
    """Attested results returned by a worker after evaluating an OptimizationWorkBundle."""

    bundle_id: str
    bundle_sha256: str
    island_id: str
    results: tuple[dict[str, Any], ...]
    elapsed_ms: float
    result_version: str = BUNDLE_RESULT_SCHEMA_VERSION
    result_sha256: str = ""

    def __post_init__(self) -> None:
        if not self.result_sha256:
            digest = hashlib.sha256(canonical_json_bytes(self._hashable_dict())).hexdigest()
            object.__setattr__(self, "result_sha256", digest)

    def _hashable_dict(self) -> dict[str, Any]:
        return {
            "result_version": self.result_version,
            "bundle_id": self.bundle_id,
            "bundle_sha256": self.bundle_sha256,
            "island_id": self.island_id,
            "results": [serializable(r) for r in self.results],
            "elapsed_ms": self.elapsed_ms,
        }

    def to_dict(self) -> dict[str, Any]:
        data = self._hashable_dict()
        data["result_sha256"] = self.result_sha256
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OptimizationBundleResult:
        if data.get("result_version") != BUNDLE_RESULT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported optimization result version: {data.get('result_version')!r}"
            )
        supplied_hash = str(data.get("result_sha256", ""))
        if not supplied_hash:
            raise ValueError("bundle result is missing result_sha256 attestation")
        result = cls(
            bundle_id=str(data["bundle_id"]),
            bundle_sha256=str(data["bundle_sha256"]),
            island_id=str(data["island_id"]),
            results=tuple(dict(r) for r in data.get("results", ())),
            elapsed_ms=float(data.get("elapsed_ms", 0.0)),
            result_version=str(data.get("result_version", BUNDLE_RESULT_SCHEMA_VERSION)),
            result_sha256=supplied_hash,
        )
        expected_hash = hashlib.sha256(canonical_json_bytes(result._hashable_dict())).hexdigest()
        if supplied_hash != expected_hash:
            raise ValueError(f"bundle result {result.bundle_id} hash mismatch")
        return result

    def to_json(self, path: Path | str) -> Path:
        return atomic_write_json(path, self.to_dict())

    @classmethod
    def from_json(cls, path: Path | str) -> OptimizationBundleResult:
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def create_work_bundle(
    *,
    run_id: str,
    optimization_id: str,
    island_id: str,
    step: int,
    profile: Profile,
    pair_spec: PairSpec,
    scenarios: Sequence[ScenarioSpec],
    evaluator_identity: Mapping[str, Any],
    yb_settings: Mapping[str, Any],
    score_key: str,
    proposals: Sequence[Mapping[str, Any]],
    pool_overrides: Mapping[str, Any] | None = None,
) -> OptimizationWorkBundle:
    """Create a frozen, content-hashed OptimizationWorkBundle."""
    bundle_id = f"bundle_{run_id}_{island_id}_{step:06d}"
    prop_dicts = []
    for idx, proposal in enumerate(proposals):
        q_params = quantized(profile, proposal["params"])
        prop_policy_params, prop_pool_overrides = split_request(profile, q_params)
        prop_dicts.append(
            {
                "ordinal": int(proposal.get("ordinal", idx)),
                "ask_id": str(proposal["ask_id"]),
                "params": q_params,
                "policy_params": prop_policy_params,
                "pool_overrides": prop_pool_overrides,
            }
        )
    scen_dicts = tuple(s.to_dict() for s in scenarios)

    return OptimizationWorkBundle(
        bundle_id=bundle_id,
        run_id=run_id,
        optimization_id=optimization_id,
        island_id=island_id,
        step=step,
        policy_id=profile.name,
        policy_header=profile.header_file,
        policy_source_sha256=profile.source_sha256,
        policy_abi=profile.policy_abi,
        policy_parameter_count=profile.n_params(),
        pool_overrides=dict(pool_overrides or {}),
        pair_spec=pair_spec.to_dict(),
        scenarios=scen_dicts,
        evaluator_identity=dict(evaluator_identity),
        yb_settings=dict(yb_settings),
        score_key=normalize_score_key(score_key),
        proposals=tuple(prop_dicts),
    )


def evaluate_work_bundle(
    bundle: OptimizationWorkBundle,
    client: HarnessClient,
) -> OptimizationBundleResult:
    """Worker entrypoint: evaluate an immutable OptimizationWorkBundle without mutating it."""
    # 1. Verify bundle content hash integrity
    computed_hash = hashlib.sha256(canonical_json_bytes(bundle._hashable_dict())).hexdigest()
    if bundle.bundle_sha256 and computed_hash != bundle.bundle_sha256:
        raise ValueError(
            f"Corrupt work bundle {bundle.bundle_id}: hash mismatch "
            f"expected {bundle.bundle_sha256}, calculated {computed_hash}"
        )
    scenarios = tuple(ScenarioSpec.from_dict(s_dict) for s_dict in bundle.scenarios)
    owned_pool = False
    if isinstance(client, SubprocessHarnessClient):
        original_client = client
        used_original = False

        def factory(_scenario: ScenarioSpec) -> HarnessClient:
            nonlocal used_original
            if not used_original:
                used_original = True
                return original_client
            return original_client.clone()

        client = ScenarioHarnessClient(scenarios, factory)
        owned_pool = True


    start_time = time.monotonic()
    identity = client.prepare()
    if not isinstance(identity, VerifiedEvaluator):
        raise TypeError("HarnessClient.prepare() must return a VerifiedEvaluator")
    validate_evaluator_identity(
        identity,
        expected_policy_id=bundle.policy_id,
        expected_policy_source_sha256=bundle.policy_source_sha256,
        expected_policy_abi=bundle.policy_abi,
        expected_policy_parameter_count=bundle.policy_parameter_count,
    )
    expected_binary = str(bundle.evaluator_identity.get("sha256") or bundle.evaluator_identity.get("binary_sha256") or "")
    if not expected_binary or identity.sha256.lower() != expected_binary.lower():
        raise ValueError(
            f"worker evaluator SHA-256 {identity.sha256!r} != bundle {expected_binary!r}"
        )

    # 2. Prepare candidate request payloads
    prepared = []
    for prop in bundle.proposals:
        cand_idx = prop["ordinal"]
        ask_id = prop["ask_id"]
        q_params = prop["params"]
        prepared.append({
            "ordinal": cand_idx,
            "ask_id": ask_id,
            "quant_params": q_params,
            "pool_overrides": dict(prop.get("pool_overrides") or bundle.pool_overrides),
            "policy_params": prop.get("policy_params", q_params),
            "scenario_results": [],
            "scenario_candidate_ids": {},
            "scenario_fingerprints": {},
            "primary_metrics": {},
            "evaluation_ok": True,
        })

    # 3. Evaluate across scenarios in batched calls
    require_yb = bool(bundle.yb_settings.get("require_yb", False))
    for scen in scenarios:

        client.open_session(scen)
        batch_requests = []
        for p in prepared:
            scen_cand_id = f"{bundle.run_id}_{p['ask_id']}_{scen.id}"
            p["scenario_candidate_ids"][scen.id] = scen_cand_id
            batch_requests.append(
                CandidateSpec(
                    ordinal=p["ordinal"],
                    candidate_id=scen_cand_id,
                    policy_params=p["policy_params"],
                    pool_overrides=p["pool_overrides"],
                )
            )

        batch_resp = client.evaluate_batch(
            batch_requests,
            observation=ObservationSpec(kind="summary"),
        )
        if batch_resp.status != "complete":
            raise RuntimeError(
                f"evaluator returned incomplete batch for scenario {scen.id!r}: "
                f"{batch_resp.status!r}"
            )

        results_by_ordinal = {r.ordinal: r for r in batch_resp.results}
        for p in prepared:
            res = results_by_ordinal.get(p["ordinal"])
            if res is not None:
                p["evaluation_ok"] = p["evaluation_ok"] and res.status == "ok"
                res_metrics = dict(res.metrics)
                res_metrics["ok"] = res.status == "ok"
                p["scenario_results"].append(res_metrics)
                if scen.id == bundle.scenarios[0]["id"]:
                    p["primary_metrics"] = dict(res.metrics)
                if res.economic_fingerprint:
                    p["scenario_fingerprints"][scen.id] = res.economic_fingerprint
            else:
                p["evaluation_ok"] = False
                p["scenario_results"].append({"ok": False})

    # 4. Score scenarios and compile candidate results
    candidate_results: list[dict[str, Any]] = []
    primary_scenario_id = bundle.scenarios[0]["id"] if bundle.scenarios else ""

    for p in prepared:
        score_res = score_scenarios(p["scenario_results"], require_yb=require_yb)
        obj_val = score_objective_value(score_res, bundle.score_key)
        score_res["objective_value"] = obj_val
        score_res["objective_failures"] = objective_failure_count(score_res, bundle.score_key)
        loss_val = loss_from_score(score_res)

        primary_fingerprint = p["scenario_fingerprints"].get(primary_scenario_id)
        primary_cand_id = p["scenario_candidate_ids"].get(primary_scenario_id, p["ask_id"])

        candidate_results.append({
            "ordinal": p["ordinal"],
            "ask_id": p["ask_id"],
            "params": p["quant_params"],
            "policy_params": p["policy_params"],
            "pool_overrides": p["pool_overrides"],
            "loss": loss_val,
            "objective": obj_val,
            "score_res": score_res,
            "candidate_id": primary_cand_id,
            "economic_fingerprint": primary_fingerprint,
            "scenario_candidate_ids": p["scenario_candidate_ids"],
            "scenario_fingerprints": p["scenario_fingerprints"],
            "primary_metrics": p["primary_metrics"],
            "status": "ok" if p["evaluation_ok"] else "failed",
        })

    elapsed = (time.monotonic() - start_time) * 1000.0

    result = OptimizationBundleResult(
        bundle_id=bundle.bundle_id,
        bundle_sha256=bundle.bundle_sha256,
        island_id=bundle.island_id,
        results=tuple(candidate_results),
        elapsed_ms=elapsed,
    )
    if owned_pool:
        client.close()
    return result
__all__ = [
    "WORK_BUNDLE_SCHEMA_VERSION",
    "BUNDLE_RESULT_SCHEMA_VERSION",
    "OptimizationWorkBundle",
    "OptimizationBundleResult",
    "create_work_bundle",
    "evaluate_work_bundle",
]
