"""Tests for strict discriminated run manifests."""

from __future__ import annotations
import json

from pathlib import Path

import pytest

from curve_fx_sim.artifacts.attestation import (
    resolve_attested_file,
    verify_manifest_artifacts,
)
from curve_fx_sim.artifacts.io import sha256_path
from curve_fx_sim.artifacts.manifest import (
    ManifestError,
    load_manifest,
    new_grid_manifest,
    new_optimization_manifest,
    new_shiftclick_manifest,
    write_manifest_atomic,
)
from curve_fx_sim.specs.common import SpecError


def _core() -> dict[str, object]:
    return {
        "schema_version": "curve_fx_sim_identity_v2",
        "binary": "arb_evaluator_ld",
        "sha256": "a" * 64,
        "harness_version": "1.0.0",
        "pool_version": "0.1.0",
        "policy_id": "test_policy",
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


def test_grid_manifest_lifecycle(tmp_path: Path) -> None:
    manifest = new_grid_manifest(
        run_id="grid_run_001",
        grid_id="grid_100",
        pool_count=4,
        resolved_spec={"pair_id": "chfusd"},
        resolved_axes=[{"name": "mid_fee", "values": ["0.0001", "0.0005"]}],
        pools=[{"id": f"p{i}", "ordinal": i} for i in range(4)],
        core=_core(),
    )
    assert manifest["grid"]["grid_id"] == "grid_100"
    out_file = tmp_path / "manifest.json"
    write_manifest_atomic(out_file, manifest, expected_kind="grid")
    assert load_manifest(out_file, expected_kind="grid")["run_id"] == "grid_run_001"
    with pytest.raises(ManifestError, match="does not match expected"):
        load_manifest(out_file, expected_kind="optimization")




def test_artifact_descriptor_schema_is_strict_at_write_and_load(tmp_path: Path) -> None:
    artifact = {
        "kind": "evaluation_table",
        "path": "evaluation_table.json",
        "sha256": "c" * 64,
        "bytes": 0,
    }
    manifest = new_grid_manifest(
        run_id="grid_artifacts",
        grid_id="grid_100",
        pool_count=1,
        resolved_spec={},
        resolved_axes=[],
        pools=[{"id": "p0", "ordinal": 0}],
        core=_core(),
        artifacts=[artifact],
    )
    out_file = tmp_path / "manifest.json"
    write_manifest_atomic(out_file, manifest, expected_kind="grid")

    for field, value, message in (
        ("kind", "", "kind must be a non-empty"),
        ("path", "../outside.json", "stay within"),
        ("path", "/etc/passwd", "relative"),
        ("sha256", "not-a-digest", "64-character"),
        ("bytes", -1, "non-negative"),
    ):
        malformed = dict(manifest)
        malformed["artifacts"] = [dict(artifact, **{field: value})]
        with pytest.raises(ManifestError, match=message):
            write_manifest_atomic(out_file, malformed, expected_kind="grid")

    malformed = dict(manifest)
    malformed["artifacts"] = [dict(artifact, extra="unsupported")]
    with pytest.raises(ManifestError, match="unsupported fields"):
        write_manifest_atomic(out_file, malformed, expected_kind="grid")

    malformed["artifacts"] = [dict(artifact, path="nested/../outside.json")]
    out_file.write_text(json.dumps(malformed), encoding="utf-8")
    with pytest.raises(ManifestError, match="stay within"):
        load_manifest(out_file, expected_kind="grid")



def test_attestation_hashes_bytes_and_rejects_untrusted_descriptors(tmp_path: Path) -> None:
    run_dir = tmp_path / "attested_run"
    run_dir.mkdir()
    payload = run_dir / "payload.json"
    payload.write_bytes(b"payload")
    descriptor = {
        "kind": "payload",
        "path": payload.name,
        "sha256": sha256_path(payload),
        "bytes": payload.stat().st_size,
    }
    manifest = {"run_id": run_dir.name, "artifacts": [descriptor]}
    assert verify_manifest_artifacts(manifest, run_dir=run_dir) == (payload,)

    payload.write_bytes(b"tampered")
    with pytest.raises(SpecError, match="byte size|SHA-256"):
        resolve_attested_file(descriptor, run_dir=run_dir, label="payload")

    for path in ("../payload.json", "/etc/passwd", r"..\\payload.json"):
        with pytest.raises(SpecError, match="relative|within"):
            resolve_attested_file(
                {**descriptor, "path": path},
                run_dir=run_dir,
                label="payload",
            )

    with pytest.raises(SpecError, match="unsupported fields"):
        verify_manifest_artifacts(
            {"run_id": run_dir.name, "artifacts": [{**descriptor, "extra": True}]},
            run_dir=run_dir,
        )
def test_optimization_manifest() -> None:
    manifest = new_optimization_manifest(
        run_id="opt_run_001",
        optimization_id="opt_tmrbcd",
        algorithm="tmrbcd",
        scenarios=["scenario_jan"],
        resolved_spec={"pair_id": "chfusd"},
        candidates_evaluated=50,
        best_candidate={"candidate_id": "cand_042"},
        core=_core(),
    )
    assert manifest["optimization"]["algorithm"] == "tmrbcd"


def test_shiftclick_manifest() -> None:
    manifest = new_shiftclick_manifest(
        run_id="sc_run_001",
        shiftclick_id="sc_diag_01",
        source_run_id="opt_run_001",
        selection={"kind": "best"},
        resolution="full",
        resolved_spec={"pair_id": "chfusd"},
        execution={"scope": "local"},
        core=_core(),
    )
    assert manifest["shiftclick"]["resolution"] == "full"
    assert manifest["shiftclick"]["execution"] == {"scope": "local"}


def test_invalid_manifest_rejection(tmp_path: Path) -> None:
    bad_manifest = {
        "schema_version": "fxsim_manifest_v1",
        "run_kind": "grid",
        "run_id": "bad_run",
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "resolved_spec": {},
        "core": {
            "schema_version": "curve_fx_sim_identity_v2",
            "binary": "b",
            "sha256": "invalid_hex",
        },
        "attempt_history": [],
        "artifacts": [],
        "grid": {
            "grid_id": "g1",
            "pool_count": 1,
            "pools": [{"id": "p0"}],
            "resolved_axes": [],
            "shards": [],
        },
    }
    with pytest.raises(ManifestError, match="64-character hex"):
        write_manifest_atomic(tmp_path / "bad.json", bad_manifest)
