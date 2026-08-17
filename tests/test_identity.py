"""Tests for evaluator identity, build attestations, and binary inspection."""

from __future__ import annotations

from pathlib import Path

import pytest
from curve_fx_harness_client.models import (
    EvaluatorIdentity as ProtocolEvaluatorIdentity,
    HelloFrame,
)

from curve_fx_sim.evaluation.identity import (
    VerifiedEvaluator,
    inspect_binary_identity,
    validate_evaluator_identity,
)


def _verified(*, policy_id: str, numeric_mode: str, digest: str) -> VerifiedEvaluator:
    return VerifiedEvaluator(
        path="/usr/local/bin/arb_evaluator_ld",
        hello=HelloFrame(
            evaluator_identity=ProtocolEvaluatorIdentity(
                binary_sha256=digest,
                harness_version="1.0.0",
                pool_version="0.1.0",
                policy_id=policy_id,
                policy_source_sha256="c" * 64,
                policy_abi="twocrypto_policy_v1",
                policy_parameter_count=1,
                numeric_mode=numeric_mode,
                real_type=numeric_mode,
                compiler="clang++-18",
                build_target="arb_evaluator_ld",
                ipo_enabled=False,
                native_tuning=False,
            ),
            metric_fields=["apy"],
        ),
    )


def test_evaluator_identity_serialization() -> None:
    ident = _verified(policy_id="test_policy_v1", numeric_mode="double", digest="a" * 64)
    core = ident.to_core_dict()
    assert core["schema_version"] == "curve_fx_sim_identity_v2"
    assert core["sha256"] == "a" * 64
    assert core["policy_id"] == "test_policy_v1"
    assert core["numeric_mode"] == "double"


def test_validate_evaluator_identity() -> None:
    ident = _verified(policy_id="policy_A", numeric_mode="double", digest="b" * 64)
    # Valid
    validate_evaluator_identity(ident, expected_policy_id="policy_A", expected_numeric_mode="double")

    # Mismatch policy
    with pytest.raises(ValueError, match="compiled policy"):
        validate_evaluator_identity(ident, expected_policy_id="policy_B")

    # Mismatch numeric mode
    with pytest.raises(ValueError, match="numeric mode"):
        validate_evaluator_identity(ident, expected_numeric_mode="longdouble")

def test_inspect_binary_identity_fails_closed_for_non_executable(tmp_path: Path) -> None:
    dummy_bin = tmp_path / "dummy_eval"
    dummy_bin.write_bytes(b"dummy binary contents")

    with pytest.raises(RuntimeError):
        inspect_binary_identity(dummy_bin)
