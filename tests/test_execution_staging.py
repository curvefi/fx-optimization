"""Tests for run-scoped staging, path containment, and bundle hashing."""

import json
import pytest
from pathlib import Path, PurePosixPath

from curve_fx_sim.execution.staging import (
    MarketFileEntry,
    StagingError,
    WorkBundle,
    prepare_work_bundle,
    remote_run_paths,
    scoped_remote_path,
    sha256_path,
    validate_run_id,
)


def test_validate_run_id_valid() -> None:
    assert validate_run_id("run_001") == "run_001"
    assert validate_run_id("grid-btc-2026-08") == "grid-btc-2026-08"
    assert validate_run_id("test.123_456") == "test.123_456"


def test_validate_run_id_invalid() -> None:
    with pytest.raises(StagingError):
        validate_run_id("")
    with pytest.raises(StagingError):
        validate_run_id("../escape")
    with pytest.raises(StagingError):
        validate_run_id("run/with/slashes")
    with pytest.raises(StagingError):
        validate_run_id("run with spaces")


def test_remote_run_paths() -> None:
    paths = remote_run_paths("test_run_1", remote_base="/home/user/arb")
    assert paths["root"] == PurePosixPath("/home/user/arb/runs/test_run_1")
    assert paths["shards"] == PurePosixPath("/home/user/arb/runs/test_run_1/shards")
    assert paths["results"] == PurePosixPath("/home/user/arb/runs/test_run_1/results")
    assert paths["manifest"] == PurePosixPath("/home/user/arb/runs/test_run_1/manifest.json")


def test_scoped_remote_path_valid() -> None:
    res = scoped_remote_path("run_1", "results/shard_001.json", remote_base="/home/user/arb")
    assert res == PurePosixPath("/home/user/arb/runs/run_1/results/shard_001.json")


def test_scoped_remote_path_escapes() -> None:
    with pytest.raises(StagingError, match="escapes"):
        scoped_remote_path("run_1", "../other_run", remote_base="/home/user/arb")
    with pytest.raises(StagingError, match="escapes"):
        scoped_remote_path("run_1", "/etc/passwd", remote_base="/home/user/arb")


def test_prepare_work_bundle(tmp_path: Path) -> None:
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
    assert bundle.template_local == tpl_file.resolve()
    assert bundle.template_remote == PurePosixPath("/home/user/arb/runs/bundle_run/template.json")
    assert len(bundle.market_files) == 1
    assert bundle.market_files[0].local_path == market_file.resolve()
    assert bundle.market_files[0].remote_path == PurePosixPath("/home/user/arb/runs/bundle_run/data/candles.json")
    local_manifest = json.loads(bundle.session_manifest_local.read_text(encoding="utf-8"))
    local_scenario = local_manifest["resolved_spec"]["scenario"]
    assert set(local_scenario) == {
        "id",
        "start_time",
        "end_time",
        "n_candles",
        "candle_filter",
        "market_files",
    }
    assert Path(local_scenario["market_files"][0]["path"]).is_absolute()
    remote_manifest = json.loads(
        (run_dir / "inputs" / "session_manifest.remote.json").read_text(encoding="utf-8")
    )
    remote_scenario = remote_manifest["resolved_spec"]["scenario"]
    assert "template_path" not in remote_scenario
    assert remote_scenario["market_files"][0]["sha256"] == sha256_path(market_file)
