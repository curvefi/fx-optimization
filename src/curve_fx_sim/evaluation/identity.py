"""Verification wrapper around the canonical evaluator protocol identity."""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from curve_fx_harness_client.models import (
    EvaluatorIdentity as ProtocolEvaluatorIdentity,
    HelloFrame,
)

from ..artifacts.io import sha256_path
from ..artifacts.manifest import CORE_IDENTITY_SCHEMA_VERSION

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class VerifiedEvaluator:
    """A canonical ``HelloFrame`` bound to the binary path that produced it.

    Identity fields live only in the harness client's protocol model.  This
    wrapper adds local/remote path context and manifest serialization without
    defining a second identity schema or inventing revision/build defaults.
    """

    path: str
    hello: HelloFrame

    @property
    def identity(self) -> ProtocolEvaluatorIdentity:
        return self.hello.evaluator_identity

    @property
    def sha256(self) -> str:
        return self.identity.binary_sha256

    @property
    def policy_id(self) -> str:
        return self.identity.policy_id

    @property
    def policy_source_sha256(self) -> str:
        return self.identity.policy_source_sha256

    @property
    def policy_abi(self) -> str:
        return self.identity.policy_abi

    @property
    def policy_parameter_count(self) -> int:
        return self.identity.policy_parameter_count

    @property
    def numeric_mode(self) -> str:
        return self.identity.numeric_mode

    @property
    def metric_fields(self) -> tuple[str, ...]:
        return tuple(self.hello.metric_fields)

    def to_dict(self) -> dict[str, Any]:
        """Serialize canonical protocol identity plus transport-local path."""
        return {
            "path": self.path,
            **self.identity.model_dump(),
            "capabilities": list(self.hello.capabilities),
            "metric_schema": self.hello.metric_schema,
            "metric_fields": list(self.hello.metric_fields),
        }

    def to_core_dict(self, *, binary_override: str | None = None) -> dict[str, Any]:
        """Convert the canonical identity to the strict run-manifest core block."""
        return {
            "schema_version": CORE_IDENTITY_SCHEMA_VERSION,
            "binary": binary_override or self.path,
            "sha256": self.identity.binary_sha256,
            "harness_version": self.identity.harness_version,
            "pool_version": self.identity.pool_version,
            "policy_id": self.identity.policy_id,
            "policy_source_sha256": self.identity.policy_source_sha256,
            "policy_abi": self.identity.policy_abi,
            "policy_parameter_count": self.identity.policy_parameter_count,
            "numeric_mode": self.identity.numeric_mode,
            "real_type": self.identity.real_type,
            "compiler": self.identity.compiler,
            "build_target": self.identity.build_target,
            "metric_schema": self.hello.metric_schema,
            "metric_fields": list(self.hello.metric_fields),
        }


def verified_evaluator_from_payload(
    payload: Mapping[str, Any],
    *,
    path: str | os.PathLike[str],
) -> VerifiedEvaluator:
    """Validate one exact protocol hello payload without compatibility aliases."""
    hello = HelloFrame.model_validate(payload)
    digest = hello.evaluator_identity.binary_sha256
    if not _SHA256_RE.fullmatch(digest):
        raise ValueError("evaluator identity binary_sha256 must be a lowercase SHA-256 digest")
    if not hello.metric_fields or any(not field for field in hello.metric_fields):
        raise ValueError("evaluator identity omitted non-empty metric_fields")
    return VerifiedEvaluator(path=str(path), hello=hello)


def inspect_binary_identity(binary_path: str | os.PathLike[str]) -> VerifiedEvaluator:
    """Execute a local evaluator and bind its canonical hello to its file digest."""
    path = Path(binary_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"evaluator binary not found: {path}")

    computed_digest = sha256_path(path)
    try:
        proc = subprocess.run(
            [str(path), "--identity-json"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"failed to query evaluator binary identity (--identity-json): {exc.stderr.strip()}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"cannot execute evaluator binary at {path}: {exc}") from exc

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"evaluator binary emitted malformed JSON for --identity-json: {proc.stdout!r}"
        ) from exc
    if not isinstance(data, Mapping):
        raise ValueError("evaluator identity JSON must be an object")
    verified = verified_evaluator_from_payload(data, path=path)
    if verified.sha256 != computed_digest:
        raise ValueError(
            f"evaluator binary reported sha256 {verified.sha256!r} "
            f"!= computed file digest {computed_digest!r}"
        )
    return verified


def validate_evaluator_identity(
    evaluator: VerifiedEvaluator,
    *,
    expected_policy_id: str | None = None,
    expected_policy_source_sha256: str | None = None,
    expected_policy_abi: str | None = None,
    expected_policy_parameter_count: int | None = None,
    expected_numeric_mode: str | None = None,
) -> None:
    """Verify canonical evaluator identity against the selected policy contract."""
    identity = evaluator.identity
    if expected_policy_id and identity.policy_id != expected_policy_id:
        raise ValueError(
            f"evaluator compiled policy {identity.policy_id!r} != expected {expected_policy_id!r}"
        )
    if expected_policy_source_sha256 and (
        identity.policy_source_sha256.lower() != expected_policy_source_sha256.lower()
    ):
        raise ValueError(
            "evaluator compiled policy source SHA-256 "
            f"{identity.policy_source_sha256!r} != expected {expected_policy_source_sha256!r}"
        )
    if expected_policy_abi and identity.policy_abi != expected_policy_abi:
        raise ValueError(
            f"evaluator compiled policy ABI {identity.policy_abi!r} != expected {expected_policy_abi!r}"
        )
    if (
        expected_policy_parameter_count is not None
        and identity.policy_parameter_count != expected_policy_parameter_count
    ):
        raise ValueError(
            "evaluator compiled policy parameter count "
            f"{identity.policy_parameter_count!r} != expected {expected_policy_parameter_count}"
        )
    if expected_numeric_mode and identity.numeric_mode != expected_numeric_mode:
        raise ValueError(
            f"evaluator numeric mode {identity.numeric_mode!r} != expected {expected_numeric_mode!r}"
        )


__all__ = [
    "VerifiedEvaluator",
    "inspect_binary_identity",
    "validate_evaluator_identity",
    "verified_evaluator_from_payload",
]
