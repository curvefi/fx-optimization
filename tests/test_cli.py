"""Tests for the fxsim Click command-line interface."""
import json
from types import SimpleNamespace
from pathlib import Path
from click.testing import CliRunner
from curve_fx_sim.cli import main


def test_cli_optimize_run_selects_run_root_relative_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    project = tmp_path / "project"
    run_root = tmp_path / "outputs"
    artifact = run_root / "artifacts" / "selected"
    artifact.mkdir(parents=True)
    spec = type("Spec", (), {"id": "demo"})()
    selected = object()
    calls: dict[str, object] = {}
    def load_spec(_ref, *, repository, parameter_space_authority):
        calls["authority"] = parameter_space_authority
        calls["repository"] = repository
        return spec
    monkeypatch.setattr(
        "curve_fx_sim.evaluation.selected.SelectedEvaluator.load",
        lambda path: (calls.update(artifact_path=path) or selected),
    )
    monkeypatch.setattr(
        "curve_fx_sim.specs.optimization.load_optimization_spec", load_spec
    )
    monkeypatch.setattr(
        "curve_fx_sim.optimization.run_optimization",
        lambda *args, **kwargs: (calls.update(kwargs) or {"ok": True}),
    )
    result = CliRunner().invoke(
        main,
        [
            "--project-root",
            str(project),
            "--run-root",
            str(run_root),
            "optimize",
            "run",
            "demo",
            "--artifact-dir",
            "artifacts/selected",
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls["artifact_path"] == artifact
    assert calls["authority"] == "selected_schema"
    assert calls["selected_evaluator"] is selected
    assert calls["client"] is None
    harness = tmp_path / "evaluator"
    harness.write_bytes(b"binary")
    conflict = CliRunner().invoke(
        main,
        [
            "--project-root",
            str(tmp_path),
            "optimize",
            "run",
            "demo",
            "--artifact-dir",
            str(artifact),
            "--harness",
            str(harness),
        ],
    )
    assert conflict.exit_code != 0
    assert "mutually exclusive" in conflict.output


def test_cli_named_grid_generate_then_run_uses_artifact_and_repository(
    tmp_path: Path, monkeypatch
) -> None:
    project = tmp_path / "project"
    run_root = tmp_path / "runs"
    artifact_dir = run_root / "artifacts" / "selected"
    artifact_dir.mkdir(parents=True)
    calls: dict[str, object] = {}
    pair = SimpleNamespace(id="pair")
    grid = SimpleNamespace(id="grid", pair_id="pair", policy_id=None, axes=())
    scenario = SimpleNamespace(id="scenario", pair_id="pair")
    identity = SimpleNamespace(metric_fields=("apy",))
    selected = SimpleNamespace(verified_evaluator=identity)
    monkeypatch.setattr("curve_fx_sim.specs.pair.load_pair_spec", lambda *_a, **_k: pair)
    monkeypatch.setattr("curve_fx_sim.specs.grid.load_grid_spec", lambda *_a, **_k: grid)
    monkeypatch.setattr("curve_fx_sim.specs.scenario.load_scenario_spec", lambda *_a, **_k: scenario)
    monkeypatch.setattr("curve_fx_sim.evaluation.selected.SelectedEvaluator.load", lambda path: (
        calls.update(artifact_path=path) or selected
    ))
    temporary_roots: list[Path] = []
    class Materialization:
        baseline_open_session_fields = {"n_candles": 3}
        closure = object()
        def validated(self):
            return self
    def materialize(_scenario, *, repository, manifest_root, session_id):
        calls.update(repository=repository, session_id=session_id)
        temporary_roots.append(manifest_root)
        return Materialization()
    monkeypatch.setattr(
        "curve_fx_sim.evaluation.session.LocalSessionMaterialization.from_scenario",
        materialize,
    )
    def compile_grid(*args, **kwargs):
        calls.update(compile_args=args, compile_kwargs=kwargs)
        return SimpleNamespace(
            manifest={"run_id": kwargs["run_id"]},
            manifest_path=run_root / "grid" / "manifest.json",
            run_dir=run_root / "grid",
            points=(object(),),
        )
    import curve_fx_sim.execution  # noqa: F401
    monkeypatch.setattr("curve_fx_sim.grids.runner.compile_grid_run", compile_grid)
    generated = CliRunner().invoke(
        main,
        [
            "--project-root", str(project), "--run-root", str(run_root),
            "grid", "generate", "--pair", "pair", "--grid", "grid",
            "--scenario", "scenario", "--artifact-dir", "artifacts/selected",
            "--run-id", "named-grid",
        ],
    )
    assert generated.exit_code == 0, generated.output
    assert calls["artifact_path"] == artifact_dir
    assert calls["repository"] == project.resolve()
    assert calls["compile_kwargs"]["selected_evaluator"] is selected
    assert calls["compile_kwargs"]["open_session"] == {"n_candles": 3}
    assert calls["compile_kwargs"]["scenario"] is Materialization.closure
    assert temporary_roots and not temporary_roots[0].exists()
    manifest = run_root / "grid" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps({"resolved_spec": {"candidate_compilation": {"mode": "schema_backed_named_proposals"}}}),
        encoding="utf-8",
    )
    summary = SimpleNamespace(
        status="ok", run_id="named-grid", scope="local", total_pools=1,
        duration_seconds=0.1, output_path=None, manifest_path=manifest,
    )
    backend_calls: dict[str, object] = {}
    class Backend:
        def __init__(self, **kwargs):
            backend_calls.update(init=kwargs)
        def run_grid(self, path, **kwargs):
            backend_calls.update(path=path, run=kwargs)
            return summary
    monkeypatch.setattr("curve_fx_sim.execution.ExecutionBackend", Backend)
    monkeypatch.setattr("curve_fx_sim.execution.load_site_profile", lambda *a, **k: object())
    ran = CliRunner().invoke(
        main,
        ["--project-root", str(project), "grid", "run", str(manifest)],
    )
    assert ran.exit_code == 0, ran.output
    assert backend_calls["run"]["repository"] == project.resolve()
