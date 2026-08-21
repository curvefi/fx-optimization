from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from curve_fx_sim.artifacts.io import sha256_path
from curve_fx_sim.evaluation import builder
from curve_fx_sim.evaluation.builder import (
    BuildSpec,
    EvaluatorBuildError,
    build_evaluator,
    build_receipt,
    load_evaluator_artifact,
)


def _sources(root: Path) -> tuple[Path, Path, Path]:
    pool = root / "pool"
    harness = root / "harness"
    policy = root / "policy.hpp"
    (pool / "include/pools/twocrypto_fx/policies").mkdir(parents=True)
    (harness / "cpp/src").mkdir(parents=True)
    (pool / "CMakeLists.txt").write_text("project(pool)\n", encoding="utf-8")
    (harness / "CMakeLists.txt").write_text("project(harness)\n", encoding="utf-8")
    passthrough = pool / "include/pools/twocrypto_fx/policies/compiled_passthrough.hpp"
    passthrough.write_text("// zero fallback\n", encoding="utf-8")
    (pool / "include/pool.hpp").write_text("// pool\n", encoding="utf-8")
    (harness / "cpp/src/main.cpp").write_text("// harness\n", encoding="utf-8")
    policy.write_text("// policy one\n", encoding="utf-8")
    return pool, harness, policy


def _spec(pool: Path, harness: Path, policy: Path | None = None, **changes: object) -> BuildSpec:
    values: dict[str, object] = {"pool_root": pool, "harness_root": harness}
    if policy is not None:
        values.update(
            policy_header=policy,
            policy_id="meaningful",
            policy_expected_sha256=sha256_path(policy),
        )
    values.update(changes)
    return BuildSpec(**values)  # type: ignore[arg-type]


def test_build_receipt_is_path_independent_and_changes_with_sources_and_mode(
    tmp_path: Path,
) -> None:
    pool_a, harness_a, policy_a = _sources(tmp_path / "a")
    pool_b, harness_b, policy_b = _sources(tmp_path / "elsewhere")

    first = build_receipt(_spec(pool_a, harness_a, policy_a, numeric_mode="f64"))
    relocated = build_receipt(_spec(pool_b, harness_b, policy_b, numeric_mode="f64"))
    assert first == relocated
    assert str(tmp_path) not in json.dumps(first)

    different_mode = build_receipt(
        _spec(pool_a, harness_a, policy_a, numeric_mode="longdouble")
    )
    assert different_mode["build_spec_sha256"] != first["build_spec_sha256"]
    assert different_mode["source_closures"] == first["source_closures"]

    policy_a.write_text("// policy two\n", encoding="utf-8")
    changed_policy = build_receipt(_spec(pool_a, harness_a, policy_a, numeric_mode="f64"))
    assert changed_policy["source_closures"]["policy"]["sha256"] != first[
        "source_closures"
    ]["policy"]["sha256"]
    assert changed_policy["build_spec_sha256"] != first["build_spec_sha256"]

    with pytest.raises(ValueError, match="exactly 'twocrypto_policy_v1'"):
        BuildSpec(pool_root=pool_a, harness_root=harness_a, policy_abi="future_v2")


def _description(binary: Path, *, policy_sha: str) -> dict[str, object]:
    content = binary.read_text(encoding="utf-8")
    target = "arb_evaluator_f64" if "arb_evaluator_f64" in content else "arb_evaluator_ld"
    mode = "double" if target.endswith("f64") else "longdouble"
    schema = {"schema_version": "curve_fx_parameter_schema_v1", "parameters": []}
    canonical_schema = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": "curve_fx_evaluator_description_v1",
        "binary_sha256": sha256_path(binary),
        "harness": {"version": "1.0.0", "revision": "test", "dirty": False},
        "pool": {"version": "0.1.0", "revision": "test", "dirty": False},
        "policy": {
            "id": "native_passthrough",
            "abi": "twocrypto_policy_v1",
            "source_sha256": policy_sha,
            "parameter_count": 0,
            "descriptor_abi_version": 1,
        },
        "build": {
            "type": "Release",
            "compiler": "test",
            "target": target,
            "numeric_mode": mode,
            "real_type": "double",
            "real_digits": 53,
            "real_max_digits10": 17,
            "wire_real_type": "IEEE-754 binary64",
            "wire_real_digits": 53,
            "ipo_enabled": False,
            "native_tuning": False,
        },
        "parameter_schema_sha256": hashlib.sha256(
            canonical_schema.encode()
        ).hexdigest(),
        "parameter_schema_canonical_json": canonical_schema,
        "parameter_schema": schema,
    }


def test_build_publishes_fresh_self_verifying_artifact_and_rejects_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pool, harness, _ = _sources(tmp_path / "sources")
    policy_sha = sha256_path(
        pool / "include/pools/twocrypto_fx/policies/compiled_passthrough.hpp"
    )

    def fake_run(argv: list[str], *, timeout: int = 1800) -> str:
        del timeout
        if argv[0] == "cmake" and "--build" not in argv:
            return ""
        if argv[0] == "cmake":
            target = argv[argv.index("--target") + 1]
            if target.startswith("arb_evaluator_"):
                binary = Path(argv[argv.index("--build") + 1]) / target
                binary.parent.mkdir(parents=True, exist_ok=True)
                binary.write_text(f"fake {target}\n", encoding="utf-8")
                binary.chmod(0o755)
            return ""
        binary = Path(argv[0])
        assert argv[1:] == ["--describe-json"]
        return json.dumps(_description(binary, policy_sha=policy_sha))

    monkeypatch.setattr(builder, "_run", fake_run)
    artifact_dir = tmp_path / "artifact"
    artifact = build_evaluator(_spec(pool, harness, numeric_mode="f64"), artifact_dir)
    assert artifact.binary_path == artifact_dir / "evaluator"
    assert artifact.binary_sha256 == sha256_path(artifact.binary_path)
    assert artifact.description["build"]["numeric_mode"] == "double"
    assert len(artifact.pool_closure.files) == 3
    assert load_evaluator_artifact(artifact_dir).artifact_sha256 == artifact.artifact_sha256

    drifted = json.loads(json.dumps(artifact.description))
    drifted["parameter_schema"]["parameters"].append({"name": "pool.drift"})
    with pytest.raises(EvaluatorBuildError, match="does not match schema"):
        builder._validate_description(
            drifted,
            artifact.binary_path,
            build_receipt(_spec(pool, harness, numeric_mode="f64")),
        )
    wrong_digest = json.loads(json.dumps(artifact.description))
    wrong_digest["parameter_schema_canonical_json"] += " "
    with pytest.raises(EvaluatorBuildError, match="sha256 does not match"):
        builder._validate_description(
            wrong_digest,
            artifact.binary_path,
            build_receipt(_spec(pool, harness, numeric_mode="f64")),
        )

    original_binary = artifact.binary_path.read_bytes()
    artifact.binary_path.write_bytes(original_binary + b"corrupt")
    with pytest.raises(EvaluatorBuildError, match="binary SHA-256"):
        load_evaluator_artifact(artifact_dir)
    artifact.binary_path.write_bytes(original_binary)
    receipt = json.loads(artifact.receipt_path.read_text(encoding="utf-8"))
    receipt["description"]["pool"]["version"] = "corrupt"
    artifact.receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(EvaluatorBuildError, match="receipt SHA-256"):
        load_evaluator_artifact(artifact_dir)

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        build_evaluator(_spec(pool, harness), existing)


def test_source_closure_rejects_symlinks(tmp_path: Path) -> None:
    pool, harness, _ = _sources(tmp_path / "sources")
    target = pool / "outside.hpp"
    target.write_text("// outside\n", encoding="utf-8")
    link = pool / "include/link.hpp"
    try:
        os.symlink(target, link)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(EvaluatorBuildError, match="unsafe"):
        build_receipt(_spec(pool, harness))
