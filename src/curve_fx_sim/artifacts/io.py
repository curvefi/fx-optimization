"""Deterministic atomic JSON publication and file hashing utilities."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from ..specs.common import canonical_json_bytes, canonical_primitive


def sha256_path(path: str | os.PathLike[str]) -> str:
    """Return the SHA-256 digest of a regular file's bytes."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"file not found: {file_path}")
    if file_path.is_symlink():
        raise ValueError(f"refusing to hash symlink: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"expected regular file: {file_path}")

    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
def atomic_write_bytes(path: str | os.PathLike[str], data: bytes) -> Path:
    """Write raw bytes using a same-directory temporary file and atomic replace."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_file = tempfile.NamedTemporaryFile(
        mode="wb",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    )
    try:
        temp_file.write(data)
        temp_file.flush()
        os.fsync(temp_file.fileno())
        temp_file.close()
        os.replace(temp_file.name, destination)
    except Exception:
        if os.path.exists(temp_file.name):
            try:
                os.remove(temp_file.name)
            except OSError:
                pass
        raise
    return destination



def atomic_write_json(
    path: str | os.PathLike[str],
    payload: Any,
    *,
    indent: int = 2,
) -> Path:
    """Serialize canonical deterministic JSON and atomically replace path."""
    normalized = canonical_primitive(payload)
    text = json.dumps(
        normalized,
        ensure_ascii=False,
        indent=indent,
        sort_keys=True,
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_file = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    )
    try:
        temp_file.write(text + "\n")
        temp_file.flush()
        os.fsync(temp_file.fileno())
        temp_file.close()
        os.replace(temp_file.name, destination)
    except Exception:
        if os.path.exists(temp_file.name):
            try:
                os.remove(temp_file.name)
            except OSError:
                pass
        raise
    return destination


__all__ = [
    "sha256_path",
    "canonical_json_bytes",
    "atomic_write_bytes",
    "atomic_write_json",
]
