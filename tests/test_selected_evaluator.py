from __future__ import annotations
import hashlib
import json
from dataclasses import FrozenInstanceError
from pathlib import Path
import pytest
from curve_fx_sim.evaluation import selected as selected_module
from curve_fx_sim.evaluation.builder import EvaluatorArtifact, EvaluatorBuildError, FileReceipt, SourceClosureReceipt
from curve_fx_sim.evaluation.identity import VerifiedEvaluator, verified_evaluator_from_payload
from curve_fx_sim.evaluation.selected import (
    SelectedEvaluator,
    materialize_selected_evaluator,
)


def _artifact(tmp_path: Path) -> EvaluatorArtifact:
    binary = tmp_path / "artifact/evaluator"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"evaluator")
    binary_sha = hashlib.sha256(binary.read_bytes()).hexdigest()
    (binary.parent / "artifact.json").write_text("{}", encoding="utf-8")
    policy_sha = "c" * 64
    schema = {"schema_version": "curve_fx_parameter_schema_v1", "parameters": []}
    canonical_schema = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    description = {
        "schema_version": "curve_fx_evaluator_description_v1",
        "binary_sha256": binary_sha,
        "harness": {"version": "1.0.0"},
        "pool": {"version": "0.1.0"},
        "policy": {
            "id": "native_passthrough",
            "abi": "twocrypto_policy_v1",
            "source_sha256": policy_sha,
            "parameter_count": 0,
            "descriptor_abi_version": 1,
        },
        "build": {
            "numeric_mode": "double",
            "real_type": "double",
            "compiler": "clang++-18",
            "target": "arb_evaluator_f64",
        },
        "parameter_schema_sha256": hashlib.sha256(canonical_schema.encode()).hexdigest(),
        "parameter_schema_canonical_json": canonical_schema,
        "parameter_schema": schema,
    }
    empty = SourceClosureReceipt("d" * 64, ())
    policy = SourceClosureReceipt(
        "e" * 64, (FileReceipt("policy_header", policy_sha, 1),)
    )
    return EvaluatorArtifact(
        binary_path=binary,
        binary_sha256=binary_sha,
        description=description,
        pool_closure=empty,
        harness_closure=empty,
        policy_closure=policy,
        build_spec_sha256="a" * 64,
        artifact_sha256="b" * 64,
        receipt_path=binary.parent / "artifact.json",
    )


def _verified(artifact: EvaluatorArtifact, **updates: object) -> VerifiedEvaluator:
    values: dict[str, object] = {
        "binary_sha256": artifact.binary_sha256,
        "harness_version": "1.0.0",
        "pool_version": "0.1.0",
        "policy_id": "native_passthrough",
        "policy_source_sha256": "c" * 64,
        "policy_abi": "twocrypto_policy_v1",
        "policy_parameter_count": 0,
        "numeric_mode": "double",
        "real_type": "double",
        "compiler": "clang++-18",
        "build_target": "arb_evaluator_f64",
        "ipo_enabled": False,
        "native_tuning": False,
    }
    values.update(updates)
    return verified_evaluator_from_payload(
        {
            "protocol": "curve_fx_eval_v1",
            "type": "hello",
            "version": 1,
            "evaluator_identity": values,
            "metric_fields": ["apy"],
        },
        path=artifact.binary_path,
    )


def _patch_selection(
    monkeypatch: pytest.MonkeyPatch,
    artifact: EvaluatorArtifact,
    verified: VerifiedEvaluator | None = None,
) -> None:
    monkeypatch.setattr(selected_module, "load_evaluator_artifact", lambda path: artifact)
    def inspect(path: str | Path) -> VerifiedEvaluator:
        assert Path(path) == artifact.binary_path
        return verified or _verified(artifact)
    monkeypatch.setattr(selected_module, "inspect_binary_identity", inspect)


def test_selection_binds_compiler_and_detached_provenance_to_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = _artifact(tmp_path)
    _patch_selection(monkeypatch, artifact)
    selected = SelectedEvaluator.load(tmp_path / "artifact")
    assert selected.compiler.schema.sha256 == selected.parameter_schema_sha256
    assert selected.compiler.schema.policy_id == selected.policy_identity["id"]
    assert selected.binary_path == artifact.binary_path
    assert selected.verified_evaluator.sha256 == artifact.binary_sha256
    selected.verified_evaluator.hello.metric_fields.append("detached-mutation")
    core = selected.manifest_core(binary_override="/shared/evaluator")
    core["metric_fields"].append("mutated")
    assert selected.manifest_core(binary_override="/shared/evaluator")["metric_fields"] == ["apy"]
    assert selected.manifest_core(binary_override="/shared/evaluator")["binary"] == "/shared/evaluator"
    assert str(tmp_path) not in selected.provenance_json
    assert selected.provenance_json == SelectedEvaluator.load(tmp_path / "elsewhere").provenance_json
    artifact.description["policy"]["id"] = "mutated"  # type: ignore[index]
    provenance = selected.provenance
    provenance["policy"]["id"] = "mutated"
    assert selected.policy_identity["id"] == "native_passthrough"
    with pytest.raises(TypeError):
        selected.artifact.description["policy"]["id"] = "mutated"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        selected.binary_sha256 = "0" * 64  # type: ignore[misc]


def test_materialize_selection_publishes_exact_two_file_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = _artifact(tmp_path / "source")
    _patch_selection(monkeypatch, artifact)
    selected = SelectedEvaluator.load(artifact.binary_path.parent)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    materialized = materialize_selected_evaluator(selected, run_dir)
    published = run_dir / "evaluator_artifact"
    assert {item.name for item in published.iterdir()} == {"artifact.json", "evaluator"}
    assert (published / "evaluator").read_bytes() == artifact.binary_path.read_bytes()
    assert materialized.provenance == selected.provenance
    assert materialize_selected_evaluator(selected, run_dir, resume=True).provenance == selected.provenance


def test_selection_rejects_description_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    drift_cases = (
        ("harness_version", "2.0.0", "harness.version"),
        ("policy_parameter_count", 1, "policy.parameter_count"),
        ("compiler", "gcc-15", "build.compiler"),
    )
    for field, value, description_field in drift_cases:
        artifact = _artifact(tmp_path / field)
        _patch_selection(monkeypatch, artifact, _verified(artifact, **{field: value}))
        with pytest.raises(EvaluatorBuildError, match=description_field.replace(".", r"\.")):
            SelectedEvaluator.load(artifact.binary_path.parent)
