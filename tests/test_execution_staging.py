"""Tests for run-scoped staging, path containment, and bundle hashing."""

import json
from pathlib import Path, PurePosixPath

from curve_fx_sim.execution.staging import (
    WorkBundle,
    prepare_work_bundle,
)


def _check_prepare_work_bundle(tmp_path: Path) -> None:
    run_dir = tmp_path / "bundle_run"
    run_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    tpl_file = tmp_path / "template.json"
    tpl_file.write_text("{}", encoding="utf-8")

    market_file = data_dir / "candles.json"
    market_file.write_text("[[1,2,3,4,5,6]]", encoding="utf-8")

    manifest_file = run_dir / "manifest.json"
    manifest_file.write_text(
        json.dumps({
            "run_id": "bundle_run",
            "resolved_spec": {
                "scenario": {
                    "id": "bundle-scenario",
                    "pair_id": "bundle-pair",
                    "template_path": "template.json",
                    "market_files": [{"path": "data/candles.json"}],
                },
            },
        }),
        encoding="utf-8",
    )

    bundle = prepare_work_bundle(manifest_file, root=tmp_path, remote_base="/home/user/arb")
    assert bundle.run_id == "bundle_run"
    assert bundle.manifest_local == manifest_file.resolve()
    assert bundle.manifest_remote == PurePosixPath("/home/user/arb/runs/bundle_run/manifest.json")
    assert len(bundle.market_files) == 1
    assert bundle.market_files[0].local_path == market_file.resolve()
    assert bundle.market_files[0].remote_path == PurePosixPath("/home/user/arb/runs/bundle_run/data/candles.json")
    assert bundle.scenario_closure is not None
    assert bundle.scenario_closure_sha256 == bundle.scenario_closure.sha256


def test_work_bundle_and_scenario_closure(tmp_path: Path) -> None:
    """Keep the staging contract in one behavior-level test."""
    _check_prepare_work_bundle(tmp_path)
