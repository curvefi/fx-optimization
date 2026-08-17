"""Tests for the fxsim Click command-line interface."""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from curve_fx_sim.artifacts.io import sha256_path
from curve_fx_sim.artifacts.manifest import (
    new_grid_manifest,
    new_shiftclick_manifest,
    write_manifest_atomic,
)
from curve_fx_sim.artifacts.tables import (
    EvaluationRow,
    EvaluationTable,
    MetricProjection,
)
from curve_fx_sim.cli import main


def _core() -> dict[str, object]:
    return {
        "schema_version": "curve_fx_sim_identity_v2",
        "binary": "arb_evaluator_ld",
        "sha256": "a" * 64,
        "harness_version": "1.0.0",
        "pool_version": "0.1.0",
        "policy_id": "policy_v1",
        "policy_source_sha256": "b" * 64,
        "policy_abi": "twocrypto_policy_v1",
        "policy_parameter_count": 1,
        "numeric_mode": "double",
        "real_type": "double",
        "compiler": "clang++",
        "build_target": "arb_evaluator_ld",
        "metric_schema": "twocrypto-summary-v1",
        "metric_fields": ["apy"],
    }


def test_cli_help() -> None:
    runner = CliRunner()
    res = runner.invoke(main, ["--help"])
    assert res.exit_code == 0
    assert "Reproducible Curve FX simulation" in res.output
    assert "data" in res.output
    assert "grid" in res.output
    assert "optimize" in res.output
    assert "replay" in res.output
    assert "plot" in res.output


def test_cli_repo_audit() -> None:
    runner = CliRunner()
    res = runner.invoke(main, ["repo", "audit"])
    assert res.exit_code == 0
    data = json.loads(res.output)
    assert data["status"] == "ok"
    assert data["audit"] == "clean"


def test_cli_optimize_status_missing() -> None:
    runner = CliRunner()
    res = runner.invoke(main, ["optimize", "status", "nonexistent_run_999"])
    assert res.exit_code != 0
    assert "not found" in res.output.lower()

def test_cli_optimize_run_forwards_execution_options(tmp_path: Path, monkeypatch) -> None:
    spec_path = tmp_path / "opt.toml"
    spec_path.write_text(
        '[optimization]\nid = "demo"\npair_id = "chfusd"\n'
        'policy_id = "native_policy_dual_ema_stale_cap_v1"\nalgorithm = "tmrbcd"\n'
        'scenarios = ["smoke"]\n',
        encoding="utf-8",
    )
    spec = type("Spec", (), {"id": "demo"})()
    calls = {}

    monkeypatch.setattr(
        "curve_fx_sim.specs.optimization.load_optimization_spec",
        lambda _: spec,
    )
    monkeypatch.setattr(
        "curve_fx_sim.optimization.run_optimization",
        lambda *args, **kwargs: (calls.update(kwargs) or {"ok": True}),
    )

    runner = CliRunner()
    res = runner.invoke(
        main,
        [
            "optimize",
            "run",
            str(spec_path),
            "--run-id",
            "run-7",
            "--resume",
            "--output-root",
            str(tmp_path / "runs"),
        ],
    )
    assert res.exit_code == 0, res.output
    assert calls["run_id"] == "run-7"
    assert calls["resume"] is True


def test_cli_plot_heatmap_requires_attested_table(tmp_path: Path) -> None:
    runner = CliRunner()
    run_dir = tmp_path / "plot_run"
    run_dir.mkdir()
    res = runner.invoke(main, ["plot", "heatmap", str(run_dir), "--metric", "gamma"])
    assert res.exit_code != 0
    assert "manifest file not found" in res.output


def test_cli_plot_trajectory_requires_attested_manifest(tmp_path: Path) -> None:
    runner = CliRunner()
    diag_dir = tmp_path / "diag"
    diag_dir.mkdir()
    (diag_dir / "trace_0.npz").write_text("dummy", encoding="utf-8")
    res = runner.invoke(main, ["plot", "trajectory", str(diag_dir)])
    assert res.exit_code != 0
    assert "manifest.json" in res.output


@pytest.mark.parametrize("command", ["heatmap", "rank"])
def test_cli_table_consumers_reject_tampering(tmp_path: Path, command: str) -> None:
    run_id = f"attested_{command}"
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    table = EvaluationTable(
        [
            EvaluationRow(
                candidate_id="candidate-0",
                ordinal=0,
                coordinates={"weight": "0.5"},
                params={"vector": [0.5]},
                metrics={"apy": 0.01},
            )
        ],
        metric_projection=MetricProjection.from_fields(["apy"]),
    )
    table_path = table.to_npz(run_dir / "evaluation_table.npz")
    table_ref = {
        "path": "evaluation_table.npz",
        "sha256": sha256_path(table_path),
        "bytes": table_path.stat().st_size,
        "row_count": 1,
    }
    manifest = new_grid_manifest(
        run_id=run_id,
        grid_id="grid-one",
        pool_count=1,
        resolved_spec={},
        resolved_axes=[{"name": "weight", "values": ["0.5"]}],
        pools=[
            {
                "id": "candidate-0",
                "ordinal": 0,
                "coordinates": {"weight": "0.5"},
                "policy_params": [0.5],
                "pool_overrides": {},
            }
        ],
        core=_core(),
        table_ref=table_ref,
    )
    write_manifest_atomic(run_dir / "manifest.json", manifest, expected_kind="grid")
    table_path.write_bytes(table_path.read_bytes() + b"\n")

    args = (
        ["plot", "heatmap", str(run_dir), "--metric", "apy"]
        if command == "heatmap"
        else ["analyze", "rank", str(run_dir), "--metric", "apy"]
    )
    result = CliRunner().invoke(main, args)
    assert result.exit_code != 0
    assert "byte size" in result.output


def test_cli_verify_shiftclick_verifies_artifacts_only(tmp_path: Path) -> None:
    run_id = "shiftclick_verify_run"
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    trace_path = run_dir / "trace.json"
    trace_path.write_text("[]\n", encoding="utf-8")
    trace_ref = {
        "path": "trace.json",
        "kind": "trace",
        "sha256": sha256_path(trace_path),
        "bytes": trace_path.stat().st_size,
    }
    manifest = new_shiftclick_manifest(
        run_id=run_id,
        shiftclick_id="verify-shiftclick",
        source_run_id="source-run",
        selection={"kind": "optimizer_winner"},
        resolution="full",
        resolved_spec={},
        execution={"scope": "local"},
        core=_core(),
        artifacts=[trace_ref],
    )
    write_manifest_atomic(
        run_dir / "manifest.json",
        manifest,
        expected_kind="shiftclick",
    )

    result = CliRunner().invoke(main, ["verify", str(run_dir)])
    assert result.exit_code == 0, result.output
    assert "artifacts_verified" in result.output
    assert '"run_kind": "shiftclick"' in result.output

    # A same-size byte flip is caught by the digest check.
    raw = trace_path.read_bytes()
    trace_path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 0xFF]))
    result = CliRunner().invoke(main, ["verify", str(run_dir)])
    assert result.exit_code != 0
    assert "SHA-256" in result.output


def test_cli_trajectory_rejects_tampered_trace(tmp_path: Path) -> None:
    run_id = "shiftclick_attested_trace"
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    trace_path = run_dir / "trace.json"
    trace_path.write_text("[]\n", encoding="utf-8")
    trace_ref = {
        "path": "trace.json",
        "kind": "trace",
        "sha256": sha256_path(trace_path),
        "bytes": trace_path.stat().st_size,
    }
    manifest = new_shiftclick_manifest(
        run_id=run_id,
        shiftclick_id="attested-trace",
        source_run_id="source-run",
        selection={"kind": "optimizer_winner"},
        resolution="full",
        resolved_spec={},
        execution={"scope": "local"},
        core=_core(),
        artifacts=[trace_ref],
    )
    write_manifest_atomic(
        run_dir / "manifest.json",
        manifest,
        expected_kind="shiftclick",
    )
    trace_path.write_bytes(trace_path.read_bytes() + b"\n")

    result = CliRunner().invoke(main, ["plot", "trajectory", str(run_dir)])
    assert result.exit_code != 0
    assert "byte size" in result.output
