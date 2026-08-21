"""Strict versioned run manifests for grid, optimization, and shiftclick runs."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..specs.common import canonical_json_bytes, canonical_primitive
from .io import atomic_write_json

SCHEMA_VERSION = "fxsim_manifest_v2"
CORE_IDENTITY_SCHEMA_VERSION = "curve_fx_sim_identity_v2"
_MANIFEST_KINDS = frozenset({"grid", "optimization", "shiftclick"})
_REQUIRED_COMMON = ("schema_version", "run_kind", "run_id", "created_at", "updated_at", "resolved_spec", "core", "attempt_history", "artifacts")
_ARTIFACT_DESCRIPTOR_FIELDS = frozenset({"kind", "path", "sha256", "bytes"})
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class ManifestError(ValueError):
    """Raised for unsupported, malformed, or mismatched manifests."""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"manifest {key} must be a non-empty string")
    return value


def _path_safe_id(value: str, key: str) -> None:
    if any(char in value for char in "/\\") or value in {".", "..", "latest"}:
        raise ManifestError(f"manifest {key} is not an immutable path-safe id: {value!r}")

def _validate_sha256(value: Any, key: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ManifestError(f"manifest {key} must be a 64-character hex SHA-256 digest, got {value!r}")


def validate_artifact_descriptor(
    descriptor: Mapping[str, Any],
    *,
    label: str = "artifact",
) -> dict[str, Any]:
    """Validate and copy one canonical, repository-contained artifact record."""
    if not isinstance(descriptor, Mapping):
        raise ManifestError(f"{label} descriptor must be an object")
    fields = set(descriptor)
    missing = sorted(_ARTIFACT_DESCRIPTOR_FIELDS - fields)
    if missing:
        raise ManifestError(
            f"{label} descriptor is missing required fields: {', '.join(missing)}"
        )
    unknown = sorted(fields - _ARTIFACT_DESCRIPTOR_FIELDS)
    if unknown:
        raise ManifestError(
            f"{label} descriptor has unsupported fields: {', '.join(unknown)}"
        )

    kind = descriptor["kind"]
    if not isinstance(kind, str) or not kind.strip():
        raise ManifestError(f"{label} descriptor kind must be a non-empty string")

    raw_path = descriptor["path"]
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ManifestError(f"{label} descriptor path must be a non-empty string")
    if "\x00" in raw_path:
        raise ManifestError(f"{label} descriptor path contains a NUL byte")
    # Treat both POSIX and Windows separators as path separators. This keeps
    # manifests portable and rejects traversal before any file is accessed.
    path_parts = raw_path.replace("\\", "/").split("/")
    if raw_path.startswith(("/", "\\")) or (
        len(raw_path) >= 2 and raw_path[1] == ":" and raw_path[0].isalpha()
    ):
        raise ManifestError(f"{label} descriptor path must be relative")
    if any(part == ".." for part in path_parts):
        raise ManifestError(f"{label} descriptor path must stay within the run directory")

    _validate_sha256(descriptor["sha256"], f"{label}.sha256")
    byte_count = descriptor["bytes"]
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
        raise ManifestError(f"{label} descriptor bytes must be a non-negative integer")
    return dict(descriptor)
def _validate_table_ref(value: Any, key: str) -> None:
    if value is None:
        return
    if not isinstance(value, Mapping):
        raise ManifestError(f"manifest {key} must be an object or null")
    path = value.get("path")
    if (
        not isinstance(path, str)
        or not path
        or path.startswith(("/", "\\"))
        or (len(path) >= 2 and path[1] == ":" and path[0].isalpha())
        or any(part in {"", ".", ".."} for part in path.replace("\\", "/").split("/"))
    ):
        raise ManifestError(f"manifest {key}.path must stay within the run directory")
    _validate_sha256(value.get("sha256"), f"{key}.sha256")
    if "bytes" in value and (
        isinstance(value["bytes"], bool) or not isinstance(value["bytes"], int) or value["bytes"] < 0
    ):
        raise ManifestError(f"manifest {key}.bytes must be a non-negative integer")


def _validate_core_identity(core: Mapping[str, Any]) -> None:
    if not isinstance(core, Mapping):
        raise ManifestError("manifest core must be an object")
    if core.get("schema_version") != CORE_IDENTITY_SCHEMA_VERSION:
        raise ManifestError(f"unsupported manifest core schema_version: {core.get('schema_version')!r}")
    _required_string(core, "binary")
    _validate_sha256(core.get("sha256"), "core.sha256")
    _required_string(core, "harness_version")
    _required_string(core, "pool_version")
    policy_id = _required_string(core, "policy_id")
    policy_source_sha256 = core.get("policy_source_sha256")
    if policy_id == "twocrypto_native":
        if policy_source_sha256 != "none":
            raise ManifestError(
                "manifest native core.policy_source_sha256 must be 'none'"
            )
    else:
        _validate_sha256(policy_source_sha256, "core.policy_source_sha256")
    _required_string(core, "policy_abi")
    parameter_count = core.get("policy_parameter_count")
    if isinstance(parameter_count, bool) or not isinstance(parameter_count, int) or parameter_count < 0:
        raise ManifestError("manifest core.policy_parameter_count must be a non-negative integer")
    numeric_mode = _required_string(core, "numeric_mode")
    if numeric_mode not in {"float", "double", "longdouble"}:
        raise ManifestError(f"manifest core.numeric_mode is unsupported: {numeric_mode!r}")
    _required_string(core, "real_type")
    _required_string(core, "compiler")
    _required_string(core, "build_target")
    _required_string(core, "metric_schema")
    metric_fields = core.get("metric_fields")
    if (
        not isinstance(metric_fields, Sequence)
        or isinstance(metric_fields, (str, bytes))
        or not metric_fields
        or any(not isinstance(field, str) or not field for field in metric_fields)
    ):
        raise ManifestError("manifest core.metric_fields must be a non-empty string array")
    if "remote_sha256" in core and core["remote_sha256"] is not None:
        _validate_sha256(core["remote_sha256"], "core.remote_sha256")


def _validate_artifacts(artifacts: Any) -> None:
    if not isinstance(artifacts, Sequence) or isinstance(artifacts, (str, bytes)):
        raise ManifestError("manifest artifacts must be an array")
    for idx, item in enumerate(artifacts):
        validate_artifact_descriptor(item, label=f"manifest artifact [{idx}]")

def _validate_grid_branch(manifest: Mapping[str, Any]) -> None:
    grid = manifest.get("grid")
    if not isinstance(grid, Mapping):
        raise ManifestError("grid run manifest requires a 'grid' section")
    if set(grid) != {"grid_id", "pool_count", "plan", "shards", "table_ref"}:
        raise ManifestError("grid section fields do not match fxsim_manifest_v2")
    _required_string(grid, "grid_id")
    pool_count = grid.get("pool_count")
    if not isinstance(pool_count, int) or pool_count <= 0:
        raise ManifestError("grid.pool_count must be a positive integer")
    plan = grid.get("plan")
    if not isinstance(plan, Mapping):
        raise ManifestError("grid.plan must be an object")
    if plan.get("schema_version") != "fxsim_cartesian_grid_v1":
        raise ManifestError("grid.plan has an unsupported schema_version")
    _validate_sha256(plan.get("plan_sha256"), "grid.plan.plan_sha256")
    authority = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if hashlib.sha256(canonical_json_bytes(authority)).hexdigest() != plan["plan_sha256"]:
        raise ManifestError("grid.plan.plan_sha256 does not match its canonical authority")
    if plan.get("grid_id") != grid.get("grid_id") or plan.get("pool_count") != pool_count:
        raise ManifestError("grid.plan identity or pool count differs from its grid section")
    shape = plan.get("coordinate_shape")
    if (
        not isinstance(shape, Sequence)
        or isinstance(shape, (str, bytes))
        or not shape
        or any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in shape)
    ):
        raise ManifestError("grid.plan.coordinate_shape must contain positive integers")
    product = 1
    for size in shape:
        product *= size
    if product != pool_count:
        raise ManifestError("grid.plan.coordinate_shape does not match grid.pool_count")
    if not isinstance(grid.get("shards"), Sequence):
        raise ManifestError("grid.shards must be an array")
    _validate_table_ref(grid.get("table_ref"), "grid.table_ref")


def _validate_optimization_branch(manifest: Mapping[str, Any]) -> None:
    opt = manifest.get("optimization")
    if not isinstance(opt, Mapping):
        raise ManifestError("optimization run manifest requires an 'optimization' section")
    _required_string(opt, "optimization_id")
    _required_string(opt, "algorithm")
    if not isinstance(opt.get("scenarios"), Sequence):
        raise ManifestError("optimization.scenarios must be an array")
    count = opt.get("candidates_evaluated")
    if not isinstance(count, int) or count < 0:
        raise ManifestError("optimization.candidates_evaluated must be a non-negative integer")
    _validate_table_ref(opt.get("table_ref"), "optimization.table_ref")


def _validate_shiftclick_branch(manifest: Mapping[str, Any]) -> None:
    sc = manifest.get("shiftclick")
    if not isinstance(sc, Mapping):
        raise ManifestError("shiftclick run manifest requires a 'shiftclick' section")
    unknown = sorted(
        set(sc)
        - {
            "shiftclick_id",
            "source_run_id",
            "selection",
            "resolution",
            "execution",
        }
    )
    if unknown:
        raise ManifestError(
            "unsupported shiftclick manifest fields: " + ", ".join(unknown)
        )
    _required_string(sc, "shiftclick_id")
    _required_string(sc, "source_run_id")
    if _required_string(sc, "resolution") != "full":
        raise ManifestError("shiftclick.resolution must be 'full'")
    if not isinstance(sc.get("selection"), Mapping):
        raise ManifestError("shiftclick.selection must be an object")
    execution = sc.get("execution")
    if not isinstance(execution, Mapping):
        raise ManifestError("shiftclick.execution must be an object")
    scope = _required_string(execution, "scope")
    if scope == "local":
        if set(execution) != {"scope"}:
            raise ManifestError("local shiftclick.execution accepts only scope")
    elif scope == "cluster":
        required = {"scope", "site", "blade", "remote_workspace"}
        if set(execution) != required:
            raise ManifestError(
                "cluster shiftclick.execution requires exactly scope, site, blade, "
                "and remote_workspace"
            )
        for key in ("site", "blade", "remote_workspace"):
            _required_string(execution, key)
    else:
        raise ManifestError(
            f"shiftclick.execution.scope must be 'local' or 'cluster', got {scope!r}"
        )


def _validate_manifest(payload: Mapping[str, Any], expected_kind: str | None = None) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ManifestError(f"manifest payload must be a mapping, got {type(payload).__name__}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError(f"unsupported manifest schema_version {payload.get('schema_version')!r}, expected {SCHEMA_VERSION!r}")
    for field in _REQUIRED_COMMON:
        if field not in payload:
            raise ManifestError(f"manifest missing required field: {field}")
    run_kind = _required_string(payload, "run_kind")
    if run_kind not in _MANIFEST_KINDS:
        raise ManifestError(f"unknown run_kind: {run_kind!r}")
    if expected_kind is not None and run_kind != expected_kind:
        raise ManifestError(f"manifest run_kind {run_kind!r} does not match expected {expected_kind!r}")
    run_id = _required_string(payload, "run_id")
    _path_safe_id(run_id, "run_id")
    _validate_core_identity(payload["core"])
    _validate_artifacts(payload["artifacts"])
    if not isinstance(payload["attempt_history"], Sequence):
        raise ManifestError("manifest attempt_history must be an array")
    if run_kind == "grid":
        _validate_grid_branch(payload)
    elif run_kind == "optimization":
        _validate_optimization_branch(payload)
    else:
        _validate_shiftclick_branch(payload)
    return copy.deepcopy(dict(payload))


def load_manifest(path: str | os.PathLike[str], expected_kind: str | None = None) -> dict[str, Any]:
    """Load a manifest from path and validate it strictly."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"manifest file not found: {file_path}")
    try:
        with file_path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"malformed JSON manifest {file_path}: {exc}") from exc
    return _validate_manifest(payload, expected_kind)


def write_manifest_atomic(path: str | os.PathLike[str], manifest: Mapping[str, Any], *, expected_kind: str | None = None) -> Path:
    """Validate a manifest completely, then atomically write it."""
    return atomic_write_json(path, _validate_manifest(manifest, expected_kind))


def _build_core_dict(core: Mapping[str, Any] | None) -> dict[str, Any]:
    if core is None:
        raise ManifestError("manifest construction requires an observed evaluator core identity")
    return dict(core)


def new_grid_manifest(*, run_id: str, grid_id: str, pool_count: int, resolved_spec: Mapping[str, Any], plan: Mapping[str, Any], shards: Sequence[Mapping[str, Any]] = (), core: Mapping[str, Any] | None = None, attempt_history: Sequence[Mapping[str, Any]] = (), artifacts: Sequence[Mapping[str, Any]] = (), table_ref: Mapping[str, Any] | None = None, created_at: str | None = None, updated_at: str | None = None) -> dict[str, Any]:
    now_ts = _now()
    manifest = {"schema_version": SCHEMA_VERSION, "run_kind": "grid", "run_id": run_id, "created_at": created_at or now_ts, "updated_at": updated_at or now_ts, "resolved_spec": canonical_primitive(resolved_spec), "core": _build_core_dict(core), "attempt_history": [canonical_primitive(a) for a in attempt_history], "artifacts": [canonical_primitive(a) for a in artifacts], "grid": {"grid_id": grid_id, "pool_count": pool_count, "plan": canonical_primitive(plan), "shards": [canonical_primitive(s) for s in shards], "table_ref": canonical_primitive(table_ref) if table_ref else None}}
    return _validate_manifest(manifest, "grid")


def new_optimization_manifest(*, run_id: str, optimization_id: str, algorithm: str, scenarios: Sequence[str], resolved_spec: Mapping[str, Any], candidates_evaluated: int = 0, best_candidate: Mapping[str, Any] | None = None, core: Mapping[str, Any] | None = None, attempt_history: Sequence[Mapping[str, Any]] = (), artifacts: Sequence[Mapping[str, Any]] = (), table_ref: Mapping[str, Any] | None = None, created_at: str | None = None, updated_at: str | None = None) -> dict[str, Any]:
    now_ts = _now()
    manifest = {"schema_version": SCHEMA_VERSION, "run_kind": "optimization", "run_id": run_id, "created_at": created_at or now_ts, "updated_at": updated_at or now_ts, "resolved_spec": canonical_primitive(resolved_spec), "core": _build_core_dict(core), "attempt_history": [canonical_primitive(a) for a in attempt_history], "artifacts": [canonical_primitive(a) for a in artifacts], "optimization": {"optimization_id": optimization_id, "algorithm": algorithm, "scenarios": list(scenarios), "candidates_evaluated": candidates_evaluated, "best_candidate": canonical_primitive(best_candidate) if best_candidate else None, "table_ref": canonical_primitive(table_ref) if table_ref else None}}
    return _validate_manifest(manifest, "optimization")


def new_shiftclick_manifest(*, run_id: str, shiftclick_id: str, source_run_id: str, selection: Mapping[str, Any], resolution: str, resolved_spec: Mapping[str, Any], execution: Mapping[str, Any], core: Mapping[str, Any] | None = None, attempt_history: Sequence[Mapping[str, Any]] = (), artifacts: Sequence[Mapping[str, Any]] = (), created_at: str | None = None, updated_at: str | None = None) -> dict[str, Any]:
    now_ts = _now()
    manifest = {"schema_version": SCHEMA_VERSION, "run_kind": "shiftclick", "run_id": run_id, "created_at": created_at or now_ts, "updated_at": updated_at or now_ts, "resolved_spec": canonical_primitive(resolved_spec), "core": _build_core_dict(core), "attempt_history": [canonical_primitive(a) for a in attempt_history], "artifacts": [canonical_primitive(a) for a in artifacts], "shiftclick": {"shiftclick_id": shiftclick_id, "source_run_id": source_run_id, "selection": canonical_primitive(selection), "resolution": resolution, "execution": canonical_primitive(execution)}}
    return _validate_manifest(manifest, "shiftclick")


__all__ = ["SCHEMA_VERSION", "CORE_IDENTITY_SCHEMA_VERSION", "ManifestError", "validate_artifact_descriptor", "load_manifest", "write_manifest_atomic", "new_grid_manifest", "new_optimization_manifest", "new_shiftclick_manifest"]
