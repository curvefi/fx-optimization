import hashlib, json, math; from contextlib import nullcontext
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
import pytest
import numpy as np
from curve_fx_harness_client.models import BatchResultFrame, CandidateResult, SessionReadyFrame
from curve_fx_sim.artifacts.io import sha256_path
from curve_fx_sim.artifacts.store import RunStore
from curve_fx_sim.artifacts.tables import EvaluationRow, EvaluationTable, MetricProjection
from curve_fx_sim.evaluation.plans import CandidateCompiler
from curve_fx_sim.evaluation.selected import SelectedEvaluator
from curve_fx_sim.execution.shared_nfs import execution_site_payload, package_identity_sha256
from curve_fx_sim.execution.site import ClusterConfig, SiteProfile
from curve_fx_sim.optimization.runtime import run_optimization
from curve_fx_sim.specs.common import canonical_json_bytes
from curve_fx_sim.specs.optimization import OptimizationSpec
from curve_fx_sim.specs.pair import PairSpec
from curve_fx_sim.specs.scenario import ScenarioSpec
METRICS = {"apy_net": .05, "apy_net_gm": .04, "avg_rel_price_diff": .005, "detach_energy_ungated": .001, "duration_s": 86400., "max_7d_rel_price_diff": .02, "trades": 100, "tw_real_slippage_1pct": .0005}
def _d(name, path, kind, unit, wire, classification="session", **extra):
    return {"name": name, "lowering_path": path, "type": kind, "unit": unit, "wire_representation": wire, "classification": classification, **extra}
def _compiler():
    parameters = [
        _d("policy.weight", "evaluate_batch.candidates[].policy_params[0]", "real", "unit", "finite_binary64", "candidate", order=0, default=1., minimum=0., maximum=2., quantum=.5),
        _d("pool.out_fee", "pool_overrides.pool.out_fee", "real", "fee_fraction", "binary64_fraction_or_1e10", "candidate"),
        _d("run.session_id", "open_session.session_id", "string", "identifier", "utf8"), _d("run.template_path", "open_session.template_path", "string", "path", "utf8"), _d("run.template_sha256", "open_session.template_sha256", "string", "sha256", "lower_hex_64"), _d("run.manifest_path", "open_session.manifest_path", "string", "path", "utf8"),
        _d("run.manifest_sha256", "open_session.manifest_sha256", "string", "sha256", "lower_hex_64"), _d("run.pool_index", "open_session.pool_index", "integer", "count", "uint64", default=0), _d("run.n_candles", "open_session.n_candles", "integer", "count", "uint64", default=10), _d("run.yb_mode", "open_session.yb_mode", "enum", "yb_mode", "utf8", default="off", choices=["off", "passive", "active_2l"]),
        _d("run.yb_releverage", "open_session.yb_releverage", "boolean", "legacy_alias", "json_boolean", default=False), _d("run.metric_projection", "evaluate_batch.metric_projection", "enum", "projection", "utf8", "observation", choices=["summary", "full"]),
    ]
    schema = {"schema_version": "curve_fx_parameter_schema_v1", "parameters": parameters}
    canonical = canonical_json_bytes(schema).decode(); return CandidateCompiler.from_description({"schema_version": "curve_fx_evaluator_description_v1", "policy": {"id": "compiled_named", "parameter_count": 1, "descriptor_abi_version": 1}, "parameter_schema": schema, "parameter_schema_canonical_json": canonical, "parameter_schema_sha256": hashlib.sha256(canonical.encode()).hexdigest()})
class _Selected(SelectedEvaluator):
    def __init__(self, root, compiler, publish=True):
        root.mkdir(parents=True, exist_ok=True)
        binary, receipt = root / "evaluator", root / "artifact.json"
        if publish: binary.write_bytes(b"evaluator"); receipt.write_text("{}")
        binary_sha = sha256_path(binary)
        identity = {"binary_sha256": binary_sha, "harness_version": "1", "pool_version": "1", "policy_id": "compiled_named", "policy_source_sha256": "c" * 64, "policy_abi": "twocrypto_policy_v1", "policy_parameter_count": 1, "compiler": "test", "numeric_mode": "double", "real_type": "double", "build_target": "test", "ipo_enabled": False, "native_tuning": False}
        hello = {"protocol": "curve_fx_eval_v1", "type": "hello", "version": 1, "evaluator_identity": identity, "metric_fields": list(METRICS)}
        provenance = {"schema_version": "curve_fx_selected_evaluator_v1", "artifact_sha256": "a" * 64, "build_spec_sha256": "b" * 64, "binary_sha256": binary_sha, "parameter_schema_sha256": compiler.schema.sha256, "policy": {"id": "compiled_named", "abi": "twocrypto_policy_v1", "source_sha256": "c" * 64}}
        values = {"artifact": SimpleNamespace(receipt_path=receipt), "compiler": compiler, "_verified_payload_json": canonical_json_bytes(hello).decode(), "binary_path": binary, "binary_sha256": binary_sha, "artifact_sha256": "a" * 64, "build_spec_sha256": "b" * 64, "parameter_schema_sha256": compiler.schema.sha256, "provenance_json": canonical_json_bytes(provenance).decode()}
        for name, value in values.items(): object.__setattr__(self, name, value)
class _Optimizer:
    tells, points = 0, ((0., .001, 10.), (2., .003, 20.))
    def __init__(self, **_): self.step, self.pending = 0, []
    def ask(self, count=1): self.pending = [list(x) for x in self.points[self.step:self.step + count]]; return self.pending
    def tell(self, points, losses, **_): type(self).tells += 1; self.step += len(points); self.pending = []
    def snapshot(self): return {"step": self.step}
    def restore(self, state): self.step = state["step"]
    best_params, best_loss, best_objective, best_score = list(points[0]), 0., 0., None
class _Client:
    def __init__(self, *_args, **_kwargs): self.session_id = ""
    def start(self): return SELECTED.verified_evaluator.hello
    def open_session(self, **request):
        self.session_id = request["session_id"]
        return SessionReadyFrame(request_id="o", session_id=self.session_id, scenarios=[], scenario_set_sha256="1" * 64, session_fingerprint="2" * 64, session_config_sha256="3" * 64, metric_schema_sha256="4" * 64)
    def evaluate_batch(self, candidates, **_):
        results = [CandidateResult(ordinal=row.ordinal, candidate_id=row.candidate_id, economic_fingerprint="f" * 64, metrics=METRICS) for row in candidates]
        return BatchResultFrame(request_id="b", session_id=self.session_id, status="complete", results=results, elapsed_ms=0)
    def close_session(self, _): pass
    def shutdown(self): pass
@pytest.fixture
def setup(tmp_path, monkeypatch):
    global SELECTED, SITE
    template = tmp_path / "template.json"
    template.write_text('{"pools":[{"pool":{"out_fee":"20000000"}}]}')
    scenarios = {key: ScenarioSpec(id=key, pair_id="pair", name=key, n_candles=10, template_path=Path("template.json"), template_sha256=sha256_path(template)) for key in ("s1", "s2")}
    monkeypatch.setattr(ScenarioSpec, "harness_session_config", lambda self: {"n_candles": self.n_candles, "yb_mode": self.yb_mode, "yb_releverage": self.yb_releverage})
    SELECTED = _Selected(tmp_path / "selected", _compiler())
    monkeypatch.setattr("curve_fx_sim.optimization.runtime.load_pair_spec", lambda *a, **k: PairSpec(id="pair", name="Pair", base_token="A", quote_token="B"))
    monkeypatch.setattr("curve_fx_sim.optimization.runtime.load_scenario_spec", lambda ref, **k: scenarios[ref])
    monkeypatch.setattr("curve_fx_sim.optimization.runtime.NevergradTwoPointsDEOptimizer", _Optimizer)
    monkeypatch.setattr("curve_fx_sim.execution.grouped.EvaluatorClient", _Client)
    monkeypatch.setattr(SelectedEvaluator, "load", staticmethod(lambda path: _Selected(path, SELECTED.compiler, False)))
    SITE = SiteProfile(name="cluster", site_type="ssh", cluster=ClusterConfig(coordinator="b1", transport="shared_nfs", remote_base=PurePosixPath(str(tmp_path)), remote_run_root=PurePosixPath(str(tmp_path / "runs")), repository_root=PurePosixPath(str(tmp_path)), worker_command="/bin/fxsim", blades=("b1", "b2")))
    monkeypatch.setattr("curve_fx_sim.optimization.runtime.load_site_profile", lambda *a, **k: SITE)
    monkeypatch.setattr("curve_fx_sim.optimization.runtime.shared_run_lease", lambda *a, **k: nullcontext(SimpleNamespace(token="lease")))
    monkeypatch.setattr("curve_fx_sim.optimization.runtime._inspect_remote_execution_closure", lambda _: {"package_sha256": package_identity_sha256(tmp_path), "worker_sha256": "f" * 64})
    _Optimizer.tells = 0
    spec = OptimizationSpec(id="named", pair_id="pair", policy_id="compiled_named", algorithm="nevergrad_two_points_de", scenarios=("s1", "s2"), parameter_space={"policy.weight": {"min": 0., "max": 2., "step": .5}, "pool.out_fee": {"min": .001, "max": .003, "step": .001}, "run.n_candles": [10, 20]}, optimizer_config={"budget": 2, "batch_size": 2, "seed": 1})
    return RunStore(tmp_path), spec
def test_named_batch_groups_numeric_sessions_and_preserves_order(setup, tmp_path, monkeypatch):
    store, spec = setup
    calls = []
    def dispatch(**kwargs):
        calls.append(kwargs)
        results = {row.evaluation_id: CandidateResult(ordinal=row.ordinal, candidate_id=row.evaluation_id, economic_fingerprint="f" * 64, metrics=METRICS) for row in kwargs["evaluations"]}
        return SimpleNamespace(results_by_evaluation_id=results)
    monkeypatch.setattr("curve_fx_sim.optimization.runtime.dispatch_grouped_evaluations", dispatch)
    result = run_optimization(spec, store=store, repository=tmp_path, site="cluster", blades=("b1", "b2"), selected_evaluator=SELECTED)
    lineage = [row.params["evaluation_lineage"][0] for row in result.table.rows]
    assert len({item["session_group_id"] for item in lineage}) == 2
    assert [row.candidate_id for row in result.table.rows] == [item["evaluation_id"] for item in lineage]
    assert calls[0]["evaluation_ids_by_blade"] == {"b1": ["c_00000_s1", "c_00000_s2"], "b2": ["c_00001_s1", "c_00001_s2"]}
    assert (calls[0]["request_namespace"], calls[0]["evaluator_workers"]) == ("opt_000000", 1)
    assert all(isinstance(value, float) and math.isfinite(value) for row in result.table.rows for value in row.metrics.values())
    payload = execution_site_payload(SITE, ("b1", "b2"))
    assert payload["evaluator_source"] == "run_local_artifact" and "harness_binary" not in payload and "/attack" not in json.dumps(payload)
    assert _Optimizer.tells == 1
def test_named_group_failure_commits_nothing(setup, tmp_path, monkeypatch):
    store, spec = setup
    monkeypatch.setattr("curve_fx_sim.optimization.runtime.dispatch_grouped_evaluations",
        lambda **_: (_ for _ in ()).throw(RuntimeError("group failed")))
    with pytest.raises(RuntimeError, match="group failed"):
        run_optimization(spec, store=store, repository=tmp_path, site="cluster", selected_evaluator=SELECTED)
    run_dir = store.get_run_dir("named")
    assert _Optimizer.tells == 0
    assert not (run_dir / "evaluation_journal.jsonl").exists()
    assert not (run_dir / "checkpoint.json").exists()
def test_named_resume_replays_prefix_and_rejects_changed_lineage(setup, tmp_path):
    with pytest.raises(ValueError, match="digest"):
        MetricProjection(fields=("apy_net",), projection_sha256="0" * 64)
    invalid_table = EvaluationTable([EvaluationRow("candidate", ordinal=1)], metric_projection=MetricProjection.from_fields(("apy_net",)))
    invalid_path = tmp_path / "noncontiguous.npz"
    invalid_table.to_npz(invalid_path)
    with np.load(invalid_path) as archive: payload = {name: archive[name] for name in archive.files}
    payload["ordinal"] = np.asarray([1], dtype=np.int64)
    np.savez_compressed(invalid_path, **payload)
    with pytest.raises(ValueError, match="contiguous"):
        EvaluationTable.from_npz(invalid_path)
    store, spec = setup
    first = run_optimization(spec, store=store, repository=tmp_path, run_id="resume", budget=1, batch_size=1, selected_evaluator=SELECTED)
    run_dir = first.manifest_path.parent
    first_row = first.table.rows[0].to_dict(); checkpoint_one = (run_dir / "checkpoint.json").read_bytes()
    for name in ("manifest.json", "winner.json", "topk.json"):
        (run_dir / name).unlink()
    two = run_optimization(spec, store=store, repository=tmp_path, run_id="resume", resume=True, selected_evaluator=SELECTED)
    for name in ("manifest.json", "winner.json", "topk.json"):
        (run_dir / name).unlink()
    (run_dir / "checkpoint.json").write_bytes(checkpoint_one)
    run_optimization(spec, store=store, repository=tmp_path, run_id="resume", resume=True, selected_evaluator=SELECTED)  # table is one row ahead of checkpoint
    for name in ("manifest.json", "winner.json", "topk.json", "checkpoint.json"):
        (run_dir / name).unlink()
    resumed = run_optimization(spec, store=store, repository=tmp_path, run_id="resume", resume=True, selected_evaluator=SELECTED)
    assert resumed.table.rows[0].to_dict() == first_row
    assert not (run_dir / "evaluation_journal.jsonl").exists()
    manifest = json.loads((run_dir / "manifest.json").read_text()); assert all("journal" not in item["kind"] for item in manifest["artifacts"])
    checkpoint_path = run_dir / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["step"] = len(two.table.rows) + 1
    checkpoint_path.write_text(json.dumps(checkpoint))
    with pytest.raises(ValueError, match="checkpoint is ahead"):
        run_optimization(spec, store=store, repository=tmp_path, run_id="resume", resume=True, selected_evaluator=SELECTED)
    checkpoint["step"] -= 1
    checkpoint_path.write_text(json.dumps(checkpoint))
    table = EvaluationTable.from_npz(run_dir / "evaluation_table.npz")
    table.rows[0].params["vector"][0] = 1.5
    table.to_npz(run_dir / "evaluation_table.npz")
    with pytest.raises(ValueError, match="row prefix changed"):
        run_optimization(spec, store=store, repository=tmp_path, run_id="resume", resume=True, selected_evaluator=SELECTED)
