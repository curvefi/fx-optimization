"""Tests for production data registry and manifest verification."""
import json
import pytest
from pathlib import Path
from curve_fx_sim.data import DataVerificationError, verify_data


def test_verify_smoke_dataset(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    fixtures_dir = data_dir / "fixtures"
    fixtures_dir.mkdir()
    fixture_file = fixtures_dir / "smoke.json"
    content = json.dumps([
        {"timestamp": 1000, "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.05, "volume": 100}
    ])
    fixture_file.write_text(f"[\n{content}\n]\n", encoding="utf-8")
    content_bytes = fixture_file.read_bytes()
    import hashlib
    file_sha = hashlib.sha256(content_bytes).hexdigest()
    file_size = len(content_bytes)
    manifest_file = data_dir / "manifest.toml"
    manifest_file.write_text(
        f"""
schema_version = "data_manifest_v1"
manifest_id = "test-manifest"
description = "Test data manifest"
[[datasets]]
id = "test-smoke"
path = "data/fixtures/smoke.json"
provider = "test"
storage = "git"
processing = "test"
sha256 = "{file_sha}"
byte_size = {file_size}
schema = "ohlcv_v1"
pair = "test"
""",
        encoding="utf-8",
    )
    verified = verify_data(root=tmp_path, manifest_path=manifest_file)
    assert len(verified) == 1
    assert verified[0].id == "test-smoke"
    assert verified[0].sha256 == file_sha
    assert verified[0].byte_size == file_size


def test_verify_data_sha_mismatch(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    fixture_file = data_dir / "bad.json"
    fixture_file.write_text("[ [1,2,3,4,5,6] ]", encoding="utf-8")
    manifest_file = data_dir / "manifest.toml"
    manifest_file.write_text(
        """
schema_version = "data_manifest_v1"
manifest_id = "test-manifest"
[[datasets]]
id = "test-bad"
path = "data/bad.json"
provider = "test"
sha256 = "0000000000000000000000000000000000000000000000000000000000000000"
byte_size = 17
schema = "ohlcv_v1"
pair = "test"
""",
        encoding="utf-8",
    )
    with pytest.raises(DataVerificationError, match="sha256 mismatch"):
        verify_data(root=tmp_path, manifest_path=manifest_file)
