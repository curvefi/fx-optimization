"""Immutable compiler selection bound to one verified evaluator artifact."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from ..specs.common import canonical_json_bytes
from ..artifacts.io import atomic_write_bytes
from .builder import EvaluatorArtifact, EvaluatorBuildError, load_evaluator_artifact
from .identity import (
    VerifiedEvaluator,
    inspect_binary_identity,
    verified_evaluator_from_payload,
)
from .plans import CandidateCompiler


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


@dataclass(frozen=True, slots=True, init=False)
class SelectedEvaluator:
    """One verified artifact, its description-derived compiler, and provenance."""

    artifact: EvaluatorArtifact
    compiler: CandidateCompiler
    _verified_payload_json: str
    binary_path: Path
    binary_sha256: str
    artifact_sha256: str
    build_spec_sha256: str
    parameter_schema_sha256: str
    provenance_json: str

    def __init__(self, artifact_dir: str | os.PathLike[str]) -> None:
        artifact = load_evaluator_artifact(Path(artifact_dir))
        # Detach before compiling so caller-held mappings cannot alter this selection.
        description = json.loads(canonical_json_bytes(artifact.description))
        compiler = CandidateCompiler.from_description(description)
        verified = inspect_binary_identity(artifact.binary_path)

        binary_sha = description.get("binary_sha256")
        schema_sha = description.get("parameter_schema_sha256")
        policy = description.get("policy")
        if binary_sha != artifact.binary_sha256 or verified.sha256 != artifact.binary_sha256:
            raise EvaluatorBuildError("selected artifact binary identity is inconsistent")
        if schema_sha != compiler.schema.sha256:
            raise EvaluatorBuildError("selected artifact parameter schema is inconsistent")
        if not isinstance(policy, Mapping) or policy.get("id") != compiler.schema.policy_id:
            raise EvaluatorBuildError("selected artifact policy identity is inconsistent")
        policy_identity = {
            "id": policy.get("id"),
            "abi": policy.get("abi"),
            "source_sha256": policy.get("source_sha256"),
        }
        policy_files = artifact.policy_closure.files
        if (
            any(not isinstance(value, str) or not value for value in policy_identity.values())
            or len(policy_files) != 1
            or policy_files[0].sha256 != policy_identity["source_sha256"]
        ):
            raise EvaluatorBuildError("selected artifact policy source is inconsistent")

        harness = description.get("harness")
        pool = description.get("pool")
        build = description.get("build")
        if not all(isinstance(item, Mapping) for item in (harness, pool, build)):
            raise EvaluatorBuildError("selected artifact description identity is incomplete")
        identity = verified.identity
        expected_identity = (
            ("harness.version", harness.get("version"), identity.harness_version),
            ("pool.version", pool.get("version"), identity.pool_version),
            ("policy.id", policy.get("id"), identity.policy_id),
            ("policy.source_sha256", policy.get("source_sha256"), identity.policy_source_sha256),
            ("policy.abi", policy.get("abi"), identity.policy_abi),
            ("policy.parameter_count", policy.get("parameter_count"), identity.policy_parameter_count),
            ("build.numeric_mode", build.get("numeric_mode"), identity.numeric_mode),
            ("build.real_type", build.get("real_type"), identity.real_type),
            ("build.compiler", build.get("compiler"), identity.compiler),
            ("build.target", build.get("target"), identity.build_target),
        )
        for field, described, identified in expected_identity:
            if described != identified:
                raise EvaluatorBuildError(
                    f"selected evaluator identity differs from artifact description at {field}"
                )

        provenance = {
            "schema_version": "curve_fx_selected_evaluator_v1",
            "artifact_sha256": artifact.artifact_sha256,
            "build_spec_sha256": artifact.build_spec_sha256,
            "binary_sha256": artifact.binary_sha256,
            "parameter_schema_sha256": compiler.schema.sha256,
            "policy": policy_identity,
        }
        frozen_artifact = replace(artifact, description=_freeze_json(description))
        object.__setattr__(self, "artifact", frozen_artifact)
        object.__setattr__(self, "compiler", compiler)
        object.__setattr__(
            self,
            "_verified_payload_json",
            canonical_json_bytes(verified.hello.model_dump()).decode(),
        )
        object.__setattr__(self, "binary_path", frozen_artifact.binary_path)
        object.__setattr__(self, "binary_sha256", frozen_artifact.binary_sha256)
        object.__setattr__(self, "artifact_sha256", frozen_artifact.artifact_sha256)
        object.__setattr__(self, "build_spec_sha256", frozen_artifact.build_spec_sha256)
        object.__setattr__(self, "parameter_schema_sha256", compiler.schema.sha256)
        object.__setattr__(self, "provenance_json", canonical_json_bytes(provenance).decode())

    @classmethod
    def load(cls, artifact_dir: str | os.PathLike[str]) -> SelectedEvaluator:
        return cls(artifact_dir)

    @property
    def policy_identity(self) -> dict[str, str]:
        """Return a detached policy identity record."""
        return json.loads(self.provenance_json)["policy"]

    @property
    def verified_evaluator(self) -> VerifiedEvaluator:
        """Return a detached canonical identity bound to this artifact binary."""
        return verified_evaluator_from_payload(
            json.loads(self._verified_payload_json), path=self.binary_path
        )

    @property
    def provenance(self) -> dict[str, Any]:
        """Return a detached, JSON-serializable provenance record."""
        return json.loads(self.provenance_json)

    def manifest_core(self, *, binary_override: str | None = None) -> dict[str, Any]:
        """Return a detached strict core derived from the bound protocol identity."""
        core = self.verified_evaluator.to_core_dict(binary_override=binary_override)
        return json.loads(canonical_json_bytes(core))


def materialize_selected_evaluator(
    selected: SelectedEvaluator,
    run_dir: Path,
    *,
    resume: bool = False,
) -> SelectedEvaluator:
    """Publish and reverify the two-file evaluator closure for one run."""
    if not isinstance(selected, SelectedEvaluator):
        raise TypeError("selected must be a SelectedEvaluator")
    root = Path(run_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"run directory not found: {root}")
    destination = root / "evaluator_artifact"
    expected_names = {"artifact.json", "evaluator"}
    if resume:
        if not destination.is_dir() or {
            path.name for path in destination.iterdir()
        } != expected_names:
            raise ValueError(
                "run-local evaluator artifact is missing or contains unexpected files"
            )
    else:
        if destination.exists():
            raise FileExistsError("run-local evaluator artifact already exists")
        staging = Path(tempfile.mkdtemp(prefix=".evaluator_artifact.", dir=root))
        try:
            atomic_write_bytes(
                staging / "artifact.json", selected.artifact.receipt_path.read_bytes()
            )
            copied_binary = atomic_write_bytes(
                staging / "evaluator", selected.binary_path.read_bytes()
            )
            copied_binary.chmod(selected.binary_path.stat().st_mode)
            staging.rename(destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    materialized = SelectedEvaluator.load(destination)
    if materialized.provenance != selected.provenance:
        raise ValueError("run-local evaluator artifact differs from requested selection")
    return materialized


__all__ = ["SelectedEvaluator", "materialize_selected_evaluator"]
