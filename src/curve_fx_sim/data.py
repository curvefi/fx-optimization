"""Verification of checked-in market inputs and deterministic fixtures.

Production datasets are declared in ``data/manifest.toml``, tracked via Git LFS
or plain Git (for deterministic smoke fixtures), and verified by byte count,
SHA-256 digest, and schema headers.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
import tomllib
from typing import Any, Iterable, Mapping

from .specs.common import repository_root, assert_contained_path


class DataVerificationError(ValueError):
    """Raised when one or more manifest datasets are unavailable or invalid."""


@dataclass(frozen=True)
class VerifiedDataset:
    """A manifest record whose bytes and lightweight schema were verified."""

    id: str
    pair: str
    path: Path
    schema: str
    sha256: str
    byte_size: int

    def to_dict(self) -> dict[str, Any]:
        """Convert verified record to a serializable dictionary."""
        return {
            "id": self.id,
            "pair": self.pair,
            "path": str(self.path.as_posix()),
            "schema": self.schema,
            "sha256": self.sha256,
            "byte_size": self.byte_size,
        }


def load_data_manifest(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Load ``data/manifest.toml`` and require the v1 dataset envelope."""
    manifest_path = (
        Path(path).resolve()
        if path is not None
        else (repository_root() / "data" / "manifest.toml").resolve()
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(f"data manifest not found at {manifest_path}")

    with manifest_path.open("rb") as handle:
        payload = tomllib.load(handle)

    schema_version = str(payload.get("schema_version", ""))
    if schema_version != "data_manifest_v1":
        raise DataVerificationError(
            f"unsupported schema_version {schema_version!r} in {manifest_path}; expected 'data_manifest_v1'"
        )

    datasets = payload.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise DataVerificationError(f"{manifest_path} must declare a non-empty [[datasets]] array")

    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            prefix = handle.read(64)
    except OSError:
        return False
    return prefix.startswith(b"version https://git-lfs.github.com/spec/v1\n")


def _json_schema_ok(path: Path, schema: str) -> str | None:
    """Check the first JSON row without materializing a multi-GB candle file."""
    if schema != "ohlcv_v1":
        return f"unknown json dataset schema {schema!r}"

    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("["):
                    continue
                row_text = stripped.rstrip(",")
                if not row_text or row_text == "]":
                    continue
                parsed = json.loads(row_text)
                if not isinstance(parsed, (list, tuple)) or len(parsed) < 6:
                    return f"expected 6-element OHLCV candle row in {path.name}, got {type(parsed).__name__}"
                return None
    except Exception as exc:  # noqa: BLE001
        return f"failed to read candle row from {path.name}: {exc}"

    return None


def _csv_schema_ok(path: Path, schema: str) -> str | None:
    expected_oracle = ["timestamp", "datetime_utc", "block_number"]
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if not header:
                return f"empty CSV file {path.name}"
            if schema in {"chainlink_answers_v1", "chainlink_seed_v1"} and header[:3] != expected_oracle:
                return f"CSV header mismatch in {path.name}: got {header[:3]!r}, expected {expected_oracle!r}"
            return None
    except Exception as exc:  # noqa: BLE001
        return f"failed to parse CSV header from {path.name}: {exc}"


def _schema_error(path: Path, record: Mapping[str, Any]) -> str | None:
    schema = str(record.get("schema", ""))
    if schema == "ohlcv_v1":
        return _json_schema_ok(path, schema)
    if schema in {"chainlink_answers_v1", "chainlink_seed_v1"}:
        return _csv_schema_ok(path, schema)
    return f"unsupported dataset schema {schema!r}"


def verify_data(
    root: str | os.PathLike[str] | None = None,
    manifest_path: str | os.PathLike[str] | None = None,
) -> tuple[VerifiedDataset, ...]:
    """Verify every manifest file and return immutable verified records.

    Raises :class:`DataVerificationError` with a single combined diagnostic message
    if any dataset is missing, is an un-pulled Git LFS pointer, has a checksum mismatch,
    or fails schema verification.
    """
    root_dir = repository_root(root)
    manifest = load_data_manifest(manifest_path or (root_dir / "data" / "manifest.toml"))
    raw_datasets: Iterable[Mapping[str, Any]] = manifest.get("datasets", ())

    verified: list[VerifiedDataset] = []
    errors: list[str] = []

    for raw in raw_datasets:
        dataset_id = str(raw.get("id", ""))
        rel_path = str(raw.get("path", ""))
        pair = str(raw.get("pair", ""))
        schema = str(raw.get("schema", ""))
        expected_sha = str(raw.get("sha256", "")).lower()
        expected_bytes = int(raw.get("byte_size", 0))

        if not dataset_id or not rel_path:
            errors.append(f"malformed dataset entry: {raw!r}")
            continue

        target_path = (root_dir / rel_path).resolve()
        if not target_path.exists():
            errors.append(f"dataset {dataset_id!r} missing at {rel_path}")
            continue

        if _lfs_pointer(target_path):
            errors.append(
                f"dataset {dataset_id!r} at {rel_path} is an un-pulled Git LFS pointer; run 'git lfs pull'"
            )
            continue

        actual_bytes = target_path.stat().st_size
        if actual_bytes != expected_bytes:
            errors.append(
                f"dataset {dataset_id!r} size mismatch: expected {expected_bytes} bytes, got {actual_bytes}"
            )
            continue

        actual_sha = _sha256(target_path).lower()
        if actual_sha != expected_sha:
            errors.append(
                f"dataset {dataset_id!r} sha256 mismatch: expected {expected_sha}, got {actual_sha}"
            )
            continue

        schema_err = _schema_error(target_path, raw)
        if schema_err:
            errors.append(f"dataset {dataset_id!r} schema check failed: {schema_err}")
            continue

        verified.append(
            VerifiedDataset(
                id=dataset_id,
                pair=pair,
                path=target_path,
                schema=schema,
                sha256=actual_sha,
                byte_size=actual_bytes,
            )
        )

    if errors:
        raise DataVerificationError("\n".join(errors))

    return tuple(verified)


verify_manifest = verify_data
validate_data = verify_data

__all__ = [
    "DataVerificationError",
    "VerifiedDataset",
    "load_data_manifest",
    "validate_data",
    "verify_data",
    "verify_manifest",
]
