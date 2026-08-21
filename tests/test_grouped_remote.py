from __future__ import annotations
import hashlib, json, shlex
from pathlib import Path
from types import SimpleNamespace
import pytest
from curve_fx_harness_client.models import BatchResultFrame, CandidateResult, HelloFrame, SessionReadyFrame
from curve_fx_sim.artifacts.io import sha256_path
from curve_fx_sim.artifacts.manifest import new_grid_manifest
from curve_fx_sim.evaluation.grouping import CompiledEvaluation, PortableCandidate, group_evaluations
from curve_fx_sim.evaluation.identity import verified_evaluator_from_payload
from curve_fx_sim.evaluation.plans import CandidateCompiler, CandidateSchema, ObservationKey, ParameterDescriptor, SessionKey
from curve_fx_sim.evaluation.selected import SelectedEvaluator
from curve_fx_sim.evaluation.session import LocalSessionMaterialization
from curve_fx_sim.execution.adapter import ProcessResult
from curve_fx_sim.execution.backend import ExecutionBackend
from curve_fx_sim.execution.grouped_dispatch import dispatch_grouped_evaluations
from curve_fx_sim.execution.grouped_remote import GroupedRemoteError, GroupedWorkReceipt, GroupedWorkRequest, execute_grouped_work
from curve_fx_sim.execution.shared_nfs import package_identity_sha256
from curve_fx_sim.execution.site import ClusterConfig, HarnessConfig, RunnerConfig, SiteProfile
from curve_fx_sim.grids.model import GridPoint
from curve_fx_sim.grids.runner import encode_session_groups
from curve_fx_sim.specs.common import canonical_json_bytes
from curve_fx_sim.specs.scenario import MarketFileRef, ScenarioSpec

def _fixture(root: Path, count: int = 2):
    (root / "template.json").write_text("{}\n"); (root / "market.json").write_text("[]\n")
    digest = lambda name: sha256_path(root / name)
    scenario = ScenarioSpec("s", "pair", "s", n_candles=2, template_path=Path("template.json"),
        template_sha256=digest("template.json"), market_files=(MarketFileRef(Path("market.json"), digest("market.json")),))
    material = LocalSessionMaterialization.from_scenario(scenario, repository=root,
        manifest_root=root / "seed", session_id="seed").validated(); baseline = material.baseline_open_session_fields
    descriptors = []
    for name, value in baseline.items():
        kind, unit, wire = (("boolean", "legacy_alias" if name == "yb_releverage" else "flag", "json_boolean")
            if isinstance(value, bool) else ("integer", "count", "uint64") if isinstance(value, int)
            else ("real", "ratio", "binary64") if isinstance(value, float) else ("enum", "yb_mode", "utf8")
            if name == "yb_mode" else ("string", "path" if name.endswith("_path") else "identifier", "utf8"))
        descriptors.append(ParameterDescriptor(f"run.{name}", f"open_session.{name}", kind, unit, wire,
            "session", default=value, choices=("off", "passive", "active_2l") if name == "yb_mode" else ()))
    schema = CandidateSchema("curve_fx_parameter_schema_v1", "b" * 64, "p", tuple(descriptors)); compiler = CandidateCompiler(schema)
    session = SessionKey.from_request(schema, baseline); raw_obs = canonical_json_bytes(
        {"version": "curve_fx_observation_key_v1", "parameter_schema_sha256": schema.sha256, "evaluate_batch": {}})
    observation = ObservationKey(raw_obs, hashlib.sha256(raw_obs).hexdigest()); candidate = canonical_json_bytes({"policy_params": [], "pool_overrides": {}})
    evaluations = tuple(CompiledEvaluation(PortableCandidate(material.scenario_key, session, (), b"{}", candidate,
        hashlib.sha256(candidate).hexdigest()), "a" * 64, observation, i, f"e{i}") for i in range(count))
    groups = group_evaluations(evaluations, artifact_sha256="a" * 64, parameter_schema=schema)
    identity = {"binary_sha256": "c" * 64, "harness_version": "1", "pool_version": "1", "policy_id": "p",
        "policy_source_sha256": "d" * 64, "policy_abi": "v1", "policy_parameter_count": 0, "numeric_mode": "double",
        "real_type": "double", "compiler": "test", "build_target": "test", "ipo_enabled": False, "native_tuning": False}
    hello = {"protocol": "curve_fx_eval_v1", "type": "hello", "version": 1, "evaluator_identity": identity, "metric_fields": ["score"]}
    verified = verified_evaluator_from_payload(hello, path=root / "evaluator")
    provenance = {"schema_version": "curve_fx_selected_evaluator_v1", "artifact_sha256": "a" * 64,
        "build_spec_sha256": "e" * 64, "binary_sha256": "c" * 64, "parameter_schema_sha256": schema.sha256,
        "policy": {"id": "p", "abi": "v1", "source_sha256": "d" * 64}}
    selected = SimpleNamespace(compiler=compiler, artifact_sha256="a" * 64, binary_sha256="c" * 64,
        binary_path=root / "evaluator", verified_evaluator=verified, provenance=provenance,
        policy_identity=provenance["policy"], parameter_schema_sha256=schema.sha256,
        manifest_core=lambda binary_override=None: verified.to_core_dict(binary_override=binary_override))
    return scenario, evaluations, groups, selected, hello

class _Client:
    def __init__(self, hello): self.hello, self.session, self.closed = hello, "", False
    def start(self): return HelloFrame.model_validate(self.hello)
    def open_session(self, **request):
        self.session = request["session_id"]; return SessionReadyFrame(request_id="r", session_id=self.session,
            scenarios=[], scenario_set_sha256="1"*64, session_fingerprint="2"*64,
            session_config_sha256="3"*64, metric_schema_sha256="4"*64)
    def evaluate_batch(self, candidates, **kwargs):
        rows = [CandidateResult(ordinal=x.ordinal, candidate_id=x.candidate_id, economic_fingerprint="f"*64, metrics={"score": 1}) for x in candidates]
        return BatchResultFrame(request_id="b", session_id=self.session, status="complete", results=rows, elapsed_ms=0)
    def close_session(self, session_id): self.closed = True
    def shutdown(self): self.closed = True

def test_worker_roundtrip_and_tamper(tmp_path: Path, monkeypatch):
    repo = tmp_path / "repo"; repo.mkdir(); scenario, evaluations, _, selected, hello = _fixture(repo, 1)
    run_root = tmp_path / "runs"; run = run_root / "run"; (run / "evaluator_artifact").mkdir(parents=True)
    request = GroupedWorkRequest("run", "request", evaluations, (scenario,), canonical_json_bytes(selected.provenance), 1, 1,
        package_identity_sha256(repo)).validated(); path = run / "grouped_requests/request.json"; path.parent.mkdir(); path.write_bytes(request.canonical_json)
    loaded = []; monkeypatch.setattr(SelectedEvaluator, "load", lambda value: loaded.append(Path(value)) or selected)
    receipt = execute_grouped_work(path, run / "grouped_receipts/request.json", remote_run_root=run_root, repository=repo,
        blade="blade-b1", client_factory=lambda selected, work: _Client(hello))
    assert loaded == [run / "evaluator_artifact"] and [x.candidate_id for x in receipt.results] == ["e0"]
    with pytest.raises(GroupedRemoteError): execute_grouped_work(path, run / "wrong.json", remote_run_root=run_root,
        repository=repo, blade="blade-b1", client_factory=lambda selected, work: _Client(hello))
    for field, value, match in (("artifact_sha256", "0"*64, "provenance"), ("chunk_size", 1.5, "integers")):
        bad = json.loads(request.canonical_json); target = bad["selected_artifact_provenance"] if field.endswith("sha256") else bad
        target[field] = value; path.write_bytes(canonical_json_bytes(bad))
        with pytest.raises(GroupedRemoteError, match=match): execute_grouped_work(path, run / "grouped_receipts/request.json",
            remote_run_root=run_root, repository=repo, blade="blade-b1", client_factory=lambda selected, work: _Client(hello))
def test_two_blade_dispatch_rejects_duplicates_and_attestation_drift(tmp_path: Path):
    repo = tmp_path / "repo"; repo.mkdir(); scenario, evaluations, groups, selected, _ = _fixture(repo)
    assignments = {"blade-b1": ("e0",), "blade-b2": ("e1",)}
    site = _site(tmp_path); mode = ["ok"]
    class SSH:
        def run_ssh(self, blade, command, **kwargs):
            assert "/attack" not in command
            argv = shlex.split(command); request = GroupedWorkRequest.from_json(Path(argv[argv.index("grouped") + 1])); out = Path(argv[argv.index("--out") + 1])
            candidate = request.evaluations[0]; candidate_id = "e0" if mode[0] == "duplicate" else candidate.evaluation_id
            att = {"session_id": f"sess_run_{groups[0].key.sha256[:12]}", "scenario_set_sha256": "1"*64,
                "session_fingerprint": ("9" if mode[0] == "drift" and blade == "blade-b2" else "2")*64,
                "session_config_sha256": "3"*64, "metric_schema_sha256": "4"*64}
            receipt = GroupedWorkReceipt(request.sha256, blade, selected.artifact_sha256, {groups[0].key.sha256: att},
                (CandidateResult(ordinal=candidate.ordinal, candidate_id=candidate_id, economic_fingerprint="f"*64, metrics={"score": 1}),), 0.1)
            out.parent.mkdir(parents=True, exist_ok=True); out.write_bytes(receipt.canonical_json); return ProcessResult(0, "", "")
    assert dispatch_grouped_evaluations(run_root=site.cluster.remote_run_root / "run", run_id="run", selected=selected,
        evaluations=evaluations, scenarios=(scenario,), evaluation_ids_by_blade=assignments, repository=repo, site=site,
        chunk_size=1, lane_count=1, request_namespace="attempt_0001", ssh=SSH()).requests == 2
    for value in ("duplicate", "drift"):
        mode[0] = value
        with pytest.raises(GroupedRemoteError): dispatch_grouped_evaluations(run_root=site.cluster.remote_run_root / "run", run_id="run",
            selected=selected, evaluations=evaluations, scenarios=(scenario,), evaluation_ids_by_blade=assignments,
            repository=repo, site=site, chunk_size=1, lane_count=1, request_namespace="attempt_0001", ssh=SSH())
def _site(root):
    base = root / "remote"; return SiteProfile("blades", "ssh", cluster=ClusterConfig(coordinator="blade-b1", transport="shared_nfs",
        remote_base=base, remote_run_root=base / "runs", repository_root=root / "repo", worker_command="/bin/fxsim", blades=("blade-b1", "blade-b2")),
        harness=HarnessConfig(remote_binary_path=Path("/attack"), chunk_size=1), runner=RunnerConfig(max_workers=2, worker_concurrency=1))
def test_backend_grouped_cluster_resume_sends_no_completed_work(tmp_path: Path, monkeypatch):
    repo = tmp_path / "repo"; repo.mkdir(); scenario, evaluations, groups, selected, _ = _fixture(repo); site = _site(tmp_path)
    run = site.cluster.remote_run_root / "run"; artifact = run / "evaluator_artifact"; artifact.mkdir(parents=True)
    (artifact / "artifact.json").write_text("{}"); (artifact / "evaluator").write_bytes(b"x")
    points = [GridPoint(i, f"e{i}", (), {}, (), {}, "", (), groups[0].key.sha256, row) for i, row in enumerate(evaluations)]
    pools = []; [pools.append((lambda row: row | {"id": row["candidate_id"]})(point.to_dict())) for point in points]
    resolved = {"scenario": scenario.to_dict(), "policy": {"id": "p", "source_sha256": "d"*64, "policy_abi": "v1"},
        "grid": {"coordinate_shape": [], "axes": []}, "metric_projection": {"fields": ["score"], "projection_id": "grid", "projection_sha256": "0"*64},
        "evaluator_artifact_selection": selected.provenance, "candidate_compilation": {"mode": "schema_grouped_v1", "policy_id": "p",
        "parameter_schema_sha256": selected.parameter_schema_sha256, "parameter_schema_version": selected.compiler.schema.schema_version,
        "groups": encode_session_groups(groups)}}
    manifest = new_grid_manifest(run_id="run", grid_id="g", pool_count=2, resolved_spec=resolved, resolved_axes=[], pools=pools,
        core=selected.manifest_core(binary_override="evaluator_artifact/evaluator")); manifest["artifacts"] = [
        {"path": f"evaluator_artifact/{name}", "kind": kind, "sha256": sha256_path(artifact / name), "bytes": (artifact / name).stat().st_size}
        for name, kind in (("artifact.json", "evaluator_artifact_receipt"), ("evaluator", "evaluator_binary"))]
    path = run / "manifest.json"; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(canonical_json_bytes(manifest))
    monkeypatch.setattr(SelectedEvaluator, "load", lambda value: selected); process = _BackendProcess(groups[0].key.sha256, selected.artifact_sha256)
    backend = ExecutionBackend(site_profile=site, process_adapter=process); first = backend.run_grid(path, repository=repo, chunk_size=1)
    second = backend.run_grid(path, repository=repo, chunk_size=1, resume=True)
    assert first.executed_shards == 2 and second.skipped_shards == 2 and process.workers == 2
class _BackendProcess:
    def __init__(self, group, artifact): self.group, self.artifact, self.workers = group, artifact, 0
    def run(self, argv, **kwargs):
        command = argv[-1]
        assert "/attack" not in command
        if " worker grouped " in f" {command} ":
            self.workers += 1; parts = shlex.split(command); request = GroupedWorkRequest.from_json(Path(parts[parts.index("grouped") + 1])); out = Path(parts[parts.index("--out") + 1]); blade = parts[parts.index("--blade") + 1]
            att = {"session_id": f"sess_run_{self.group[:12]}", "scenario_set_sha256": "1"*64, "session_fingerprint": "2"*64,
                "session_config_sha256": "3"*64, "metric_schema_sha256": "4"*64}; rows = tuple(CandidateResult(ordinal=x.ordinal,
                candidate_id=x.evaluation_id, economic_fingerprint="f"*64, metrics={"score": 1}) for x in request.evaluations)
            out.parent.mkdir(parents=True, exist_ok=True); out.write_bytes(GroupedWorkReceipt(request.sha256, blade, self.artifact, {self.group: att}, rows, .1).canonical_json)
        return ProcessResult(0, "", "")
