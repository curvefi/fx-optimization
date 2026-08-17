"""Resolution and validation for files referenced by immutable run manifests.

Loads enforce path containment, relative-path safety, byte size, and (for
tables) NPZ schema/row-count consistency. Full SHA-256 re-verification is
available on demand (``verify_digest=True``, wired to ``fxsim verify``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from ..specs.common import (
    PathContainmentError,
    SpecError,
    assert_contained_path,
)
from .io import sha256_path
from .manifest import ManifestError, validate_artifact_descriptor
from .tables import EvaluationTable




def _validated_run_dir(manifest: Mapping[str, Any], run_dir: Path) -> Path:
    root = run_dir.resolve()
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise SpecError("manifest run_id must be a non-empty string")
    if root.name != run_id:
        raise SpecError(
            f"manifest run_id {run_id!r} does not match run directory {root.name!r}"
        )
    return root


def resolve_attested_file(
    descriptor: Mapping[str, Any],
    *,
    run_dir: Path,
    label: str,
    verify_digest: bool = False,
) -> Path:
    """Resolve and verify one relative path/size reference.

    Top-level manifest artifacts are validated against the exact canonical
    descriptor schema by their callers. This lower-level function also serves
    table references, which intentionally have their own metadata fields.
    With ``verify_digest`` the full-file SHA-256 is recomputed and compared;
    normal loads keep it False so large tables are not read twice.
    """
    if not isinstance(descriptor, Mapping):
        raise SpecError(f"{label} descriptor must be an object")
    raw_path = descriptor.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise SpecError(f"{label} descriptor path must be a non-empty string")
    if "\x00" in raw_path:
        raise SpecError(f"{label} descriptor path contains a NUL byte")
    portable_path = raw_path.replace("\\", "/")
    if portable_path.startswith("/") or (
        len(portable_path) >= 2
        and portable_path[1] == ":"
        and portable_path[0].isalpha()
    ):
        raise SpecError(f"{label} path must be relative")
    if any(part == ".." for part in portable_path.split("/")):
        raise SpecError(f"{label} path must stay within the run directory")
    expected_sha256 = descriptor.get("sha256")
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(char not in "0123456789abcdefABCDEF" for char in expected_sha256)
    ):
        raise SpecError(f"{label} descriptor has no valid SHA-256")
    expected_bytes = descriptor.get("bytes")
    if (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes < 0
    ):
        raise SpecError(f"{label} descriptor has no valid byte size")

    relative_path = Path(raw_path)
    root = run_dir.resolve()
    try:
        path = assert_contained_path(root / relative_path, root)
    except PathContainmentError as exc:
        raise SpecError(f"{label} path escapes the run directory") from exc
    if not path.is_file():
        raise SpecError(f"attested {label} does not exist")
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise SpecError(
            f"{label} byte size {actual_bytes} != attested {expected_bytes}"
        )
    if verify_digest:
        actual_sha256 = sha256_path(path)
        if actual_sha256.lower() != expected_sha256.lower():
            raise SpecError(
                f"{label} SHA-256 {actual_sha256} != attested {expected_sha256}"
            )
    return path


def find_attested_artifact(
    manifest: Mapping[str, Any],
    *,
    run_dir: Path,
    kind: str,
) -> Path:
    """Resolve exactly one top-level artifact of *kind*."""
    root = _validated_run_dir(manifest, run_dir)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Sequence) or isinstance(artifacts, (str, bytes)):
        raise SpecError("manifest artifacts must be an array")
    matches: list[Mapping[str, Any]] = []
    for index, item in enumerate(artifacts):
        try:
            validated = validate_artifact_descriptor(item, label=f"artifact[{index}]")
        except ManifestError as exc:
            raise SpecError(str(exc)) from exc
        if validated["kind"] == kind:
            matches.append(validated)
    if len(matches) != 1:
        raise SpecError(f"manifest must attest exactly one {kind} artifact")
    return resolve_attested_file(matches[0], run_dir=root, label=f"{kind} artifact")


def verify_manifest_artifacts(
    manifest: Mapping[str, Any],
    *,
    run_dir: Path,
) -> tuple[Path, ...]:
    """Verify every top-level artifact descriptor in a run manifest."""
    root = _validated_run_dir(manifest, run_dir)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Sequence) or isinstance(artifacts, (str, bytes)):
        raise SpecError("manifest artifacts must be an array")
    verified: list[Path] = []
    for index, item in enumerate(artifacts):
        try:
            validated = validate_artifact_descriptor(item, label=f"artifact[{index}]")
        except ManifestError as exc:
            raise SpecError(str(exc)) from exc
        verified.append(
            resolve_attested_file(
                validated,
                run_dir=root,
                label=f"artifact[{index}]",
                verify_digest=True,
            )
        )
    return tuple(verified)


def load_attested_evaluation_table(
    manifest: Mapping[str, Any],
    *,
    run_dir: Path,
    verify_digest: bool = False,
) -> tuple[EvaluationTable, Path]:
    """Load the sole evaluation table after resolving its manifest reference.

    Normal loads verify containment, size, and NPZ schema/row count without
    re-reading the file for its digest. With ``verify_digest`` (explicit
    ``fxsim verify``) the full-file SHA-256 is recomputed and compared.
    """
    root = _validated_run_dir(manifest, run_dir)
    run_kind = manifest.get("run_kind")
    if run_kind not in {"grid", "optimization"}:
        raise SpecError(f"run {manifest.get('run_id')} has no evaluation table")
    section = manifest.get(run_kind)
    if not isinstance(section, Mapping):
        raise SpecError(f"run has no {run_kind} manifest section")
    table_ref = section.get("table_ref")
    if not isinstance(table_ref, Mapping):
        raise SpecError("run has no attested evaluation-table reference")

    table_path = resolve_attested_file(
        table_ref,
        run_dir=root,
        label="evaluation-table",
        verify_digest=verify_digest,
    )
    expected_rows = table_ref.get("row_count")
    if (
        isinstance(expected_rows, bool)
        or not isinstance(expected_rows, int)
        or expected_rows < 0
    ):
        raise SpecError("evaluation-table descriptor has no valid row count")
    table = EvaluationTable.from_npz(table_path)
    if len(table) != expected_rows:
        raise SpecError(
            f"evaluation-table row count {len(table)} != attested {expected_rows}"
        )
    return table, table_path


__all__ = [
    "find_attested_artifact",
    "load_attested_evaluation_table",
    "resolve_attested_file",
    "verify_manifest_artifacts",
]
