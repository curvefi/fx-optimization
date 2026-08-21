from __future__ import annotations
import hashlib, json, threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from curve_fx_harness_client.models import BatchResultFrame, CandidateResult, HelloFrame, SessionReadyFrame

from curve_fx_sim.artifacts.manifest import new_grid_manifest
from curve_fx_sim.evaluation.grouping import LocalSessionGroupBinding, group_evaluations
from curve_fx_sim.evaluation.identity import verified_evaluator_from_payload
from curve_fx_sim.evaluation.plans import CandidateCompiler, CandidateSchema, ParameterDescriptor
from curve_fx_sim.evaluation.session import LocalSessionTransportReceipt
from curve_fx_sim.execution.collection import CollectionError, group_request_set_sha256, is_grouped_shard_complete, validate_shards, write_shard_result
from curve_fx_sim.execution.grouped import execute_local_groups
from curve_fx_sim.grids.model import compile_grid_points
from curve_fx_sim.grids.runner import encode_session_groups, load_grouped_grid
from curve_fx_sim.specs.common import canonical_json_bytes
from curve_fx_sim.specs.grid import AxisSpec, AxisTarget, GridSpec
from curve_fx_sim.specs.scenario import ScenarioClosure

ARTIFACT, SCHEMA = "a" * 64, "b" * 64
CORE = {
    "schema_version": "curve_fx_sim_identity_v2", "binary": "evaluator", "sha256": "c" * 64,
    "harness_version": "1", "pool_version": "1", "policy_id": "p", "policy_source_sha256": "d" * 64, "policy_abi": "v1",
    "policy_parameter_count": 0, "numeric_mode": "double", "real_type": "double", "compiler": "test", "build_target": "test",
    "metric_schema": "summary", "metric_fields": ["score"],
}

def _compiled():
    schema = CandidateSchema("curve_fx_parameter_schema_v1", SCHEMA, "p", (
        ParameterDescriptor("run.session_id", "open_session.session_id", "string", "identifier", "utf8", "session", default="seed"),
        ParameterDescriptor("run.n", "open_session.n_candles", "integer", "count", "uint64", "session", default=10),
        ParameterDescriptor("run.disable", "open_session.disable_slippage_probes", "boolean", "flag", "json_boolean", "observation", default=False),
        ParameterDescriptor("run.trace", "evaluate_batch.observation.trace_interval", "integer", "count", "uint64", "observation", default=1),
    ))
    compiler = CandidateCompiler(schema)
    axes = (AxisSpec("n", values=(10, 20), targets=(AxisTarget(("run", "n"), kind="integer"),)),
            AxisSpec("trace", values=(1, 2), targets=(AxisTarget(("run", "trace"), kind="integer"),)))
    grid = GridSpec("g", "pair", axes=axes, static_overrides={"run.disable": True})
    session = {"session_id": "seed", "n_candles": 10, "disable_slippage_probes": False}
    points = compile_grid_points(grid, compiler=compiler, artifact_sha256=ARTIFACT, open_session=session, scenario=ScenarioClosure("s", "pair", "e" * 64, ()))
    groups = group_evaluations(tuple(point.evaluation for point in points), artifact_sha256=ARTIFACT, parameter_schema=schema)
    pools = []
    for point in points:
        row = point.to_dict(); row["id"] = row.pop("candidate_id"); pools.append(row)
    resolved = {"candidate_compilation": {"mode": "schema_grouped_v1", "parameter_schema_sha256": SCHEMA, "groups": encode_session_groups(groups)}}
    manifest = new_grid_manifest(run_id="run", grid_id="g", pool_count=4, resolved_spec=resolved, resolved_axes=[], pools=pools, core=CORE)
    return compiler, points, groups, manifest

def _selected(compiler, path):
    identity = {
        "binary_sha256": "c" * 64, "harness_version": "1", "pool_version": "1",
        "policy_id": "p", "policy_source_sha256": "d" * 64, "policy_abi": "v1", "policy_parameter_count": 0,
        "numeric_mode": "double", "real_type": "double", "compiler": "test", "build_target": "test", "ipo_enabled": False, "native_tuning": False,
    }
    hello = {"protocol": "curve_fx_eval_v1", "type": "hello", "version": 1, "evaluator_identity": identity, "metric_fields": ["score"]}
    selected = SimpleNamespace(compiler=compiler, artifact_sha256=ARTIFACT, binary_sha256="c" * 64,
                               binary_path=path, verified_evaluator=verified_evaluator_from_payload(hello, path=path))
    return selected, hello

class _Client:
    instances = []
    barrier = threading.Barrier(2)
    def __init__(self, hello):
        self.hello, self.session_id = hello, ""
        self.opens, self.batches, self.closed, self.stopped = 0, [], False, False
        self.instances.append(self)
    def start(self): return HelloFrame.model_validate(self.hello)
    def open_session(self, **request):
        self.opens += 1
        self.session_id = request["session_id"]
        return SessionReadyFrame(request_id="o", session_id=self.session_id, scenarios=[],
            scenario_set_sha256="1" * 64, session_fingerprint="2" * 64,
            session_config_sha256="3" * 64, metric_schema_sha256="4" * 64)
    def evaluate_batch(self, candidates, **kwargs):
        self.batches.append(len(candidates))
        self.barrier.wait(timeout=1)
        results = [CandidateResult(ordinal=row.ordinal, candidate_id=row.candidate_id,
            economic_fingerprint="f" * 64, metrics={"score": 1})
            for row in sorted(candidates, key=lambda item: item.ordinal)]
        return BatchResultFrame(request_id="b", session_id=self.session_id, status="complete", results=results, elapsed_ms=0)
    def close_session(self, session_id): self.closed = True
    def shutdown(self): self.stopped = True

def _binding(group):
    session_id = f"sess_run_{group.key.sha256[:12]}"
    receipt = LocalSessionTransportReceipt(session_id, "t", "5" * 64, "m", "6" * 64)
    request = canonical_json_bytes({"session_id": session_id})
    return LocalSessionGroupBinding(group, None, request, hashlib.sha256(request).hexdigest(), receipt)

def test_grouped_manifest_roundtrip_is_pathless_and_routes_open_observation_to_session():
    compiler, points, groups, manifest = _compiled()
    loaded, rebuilt = load_grouped_grid(manifest, parameter_schema=compiler.schema, artifact_sha256=ARTIFACT)
    assert len(groups) == 2 and [len(group.observation_groups) for group in groups] == [2, 2]
    assert groups == rebuilt and [point.candidate_id for point in loaded] == [point.candidate_id for point in points]
    assert all(group.session_key.open_session_values["run.disable"] is True for group in groups)
    assert all("policy_params" not in row for row in manifest["grid"]["pools"]) and "session_request" not in json.dumps(manifest)

def test_grouped_executor_bounds_parallel_batches_orders_results_and_cleans_up(tmp_path: Path):
    compiler, points, groups, _ = _compiled()
    selected, hello = _selected(compiler, tmp_path / "evaluator")
    _Client.instances.clear()
    _Client.barrier = threading.Barrier(2)
    ordered = [point.evaluation.evaluation_id for point in reversed(points)]
    result = execute_local_groups(selected, groups, _binding, ordered, work_dir=tmp_path,
        chunk_size=1, max_workers=2, client_factory=lambda selected, work_dir: _Client(hello))
    assert list(result.results_by_evaluation_id) == ordered and result.workers == 2
    assert len(_Client.instances) == 2
    assert all(client.opens == 1 and client.batches == [1, 1] for client in _Client.instances)
    assert all(client.closed and client.stopped for client in _Client.instances)
    assert {receipt.session_attestation["session_fingerprint"] for receipt in result.receipts_by_session_group_id.values()} == {"2" * 64}
    bad_bind = lambda group: replace(_binding(group), session_request_sha256="0" * 64)
    with pytest.raises(BaseExceptionGroup, match="session request hash mismatch"):
        execute_local_groups(selected, groups, bad_bind, ordered, work_dir=tmp_path,
            chunk_size=1, max_workers=2, client_factory=lambda selected, work_dir: _Client(hello))
    assert len(_Client.instances) == 2

def test_grouped_collection_resume_and_tamper(tmp_path: Path):
    _, points, _, manifest = _compiled()
    results = tmp_path / "results"; results.mkdir()
    shards = []
    for index, point in enumerate(points):
        group, observation = point.session_group_id, point.evaluation.observation_key.sha256
        descriptor = {"shard_id": f"shard_{index}", "shard_index": index, "blade": "local",
                      "ranges": [[index, index + 1]], "chunk_size": 1, "total_pools": 4,
                      "assigned_pools": 1, "session_group_id": group, "observation_id": observation}
        shards.append(descriptor)
        attestation = {"session_id": f"sess_run_{group[:12]}", "scenario_set_sha256": "1" * 64,
                       "session_fingerprint": "2" * 64, "session_config_sha256": "3" * 64,
                       "metric_schema_sha256": "4" * 64}
        row = {"pool_index": index, "ordinal": index, "candidate_id": point.candidate_id,
               "economic_fingerprint": "f" * 64, "metrics": {"score": 1}}
        write_shard_result(results / f"shard_{index}.json", run_id="run", shard_id=descriptor["shard_id"],
            shard_index=index, ranges=((index, index + 1),), rows=[row],
            request_set_sha256=group_request_set_sha256(manifest, [index], attestation), session_attestation=attestation)
    manifest["grid"]["shards"] = shards
    validate_shards(manifest, results)
    assert all(is_grouped_shard_complete(manifest, shard, results / f"{shard['shard_id']}.json") for shard in shards)
    for field in ("session_group_id", "observation_id"):
        tampered = json.loads(json.dumps(manifest))
        tampered["grid"]["shards"][0][field] = "0" * 64
        assert not is_grouped_shard_complete(tampered, tampered["grid"]["shards"][0], results / "shard_0.json")
        with pytest.raises(CollectionError):
            validate_shards(tampered, results)
    payload = json.loads((results / "shard_0.json").read_text())
    payload["session_attestation"]["session_config_sha256"] = "0" * 64
    (results / "shard_0.json").write_text(json.dumps(payload))
    assert not is_grouped_shard_complete(manifest, shards[0], results / "shard_0.json")
