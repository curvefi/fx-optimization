"""Tests for canonical atomic JSON publication and file hashing."""

from __future__ import annotations

import json
from pathlib import Path

from curve_fx_sim.artifacts.io import atomic_write_bytes, atomic_write_json, sha256_path


def test_sha256_hashing(tmp_path: Path) -> None:
    target = tmp_path / "test.bin"
    target.write_bytes(b"hello world 123")
    assert sha256_path(target) == "d4223bf93e202505a6a501421a88d9fa43341f7757e217dd603ccdce157c13bd"


def test_atomic_json_write(tmp_path: Path) -> None:
    json_path = tmp_path / "out.json"
    atomic_write_json(json_path, {"key": "value", "num": 42})
    with json_path.open("r", encoding="utf-8") as stream:
        assert json.load(stream) == {"key": "value", "num": 42}


def test_atomic_bytes_replacement(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    atomic_write_bytes(path, b"first")
    atomic_write_bytes(path, b"second")
    assert path.read_bytes() == b"second"
