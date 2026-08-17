"""Artifact management, manifests, common evaluation tables, and run storage."""

from .attestation import (
    find_attested_artifact,
    load_attested_evaluation_table,
    resolve_attested_file,
    verify_manifest_artifacts,
)
from .io import atomic_write_bytes, atomic_write_json, canonical_json_bytes, sha256_path
from .manifest import (
    CORE_IDENTITY_SCHEMA_VERSION,
    SCHEMA_VERSION,
    ManifestError,
    load_manifest,
    new_grid_manifest,
    new_optimization_manifest,
    new_shiftclick_manifest,
    validate_artifact_descriptor,
    write_manifest_atomic,
)
from .store import RunStore
from .tables import EvaluationRow, EvaluationTable, MetricProjection

__all__ = [
    "sha256_path",
    "canonical_json_bytes",
    "atomic_write_bytes",
    "atomic_write_json",
    "SCHEMA_VERSION",
    "validate_artifact_descriptor",
    "CORE_IDENTITY_SCHEMA_VERSION",
    "ManifestError",
    "load_manifest",
    "write_manifest_atomic",
    "new_grid_manifest",
    "new_optimization_manifest",
    "new_shiftclick_manifest",
    "MetricProjection",
    "EvaluationRow",
    "EvaluationTable",
    "RunStore",
    "find_attested_artifact",
    "load_attested_evaluation_table",
    "resolve_attested_file",
    "verify_manifest_artifacts",
]
