"""Bind prepared inputs to adjacent dataset manifests, without generating data."""

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .config import ConfigError


def dataset_metadata(inputs: Mapping[str, str]) -> dict[str, Any]:
    """Verify only the selected published files and return portable identities."""
    result = {}
    for name in ("market", "price_feed", "trade_flow"):
        if name not in inputs:
            continue
        path = Path(inputs[name])
        manifest_path = path.with_name("dataset.json")
        if not manifest_path.exists():
            continue
        try:
            encoded = manifest_path.read_bytes()
            manifest = json.loads(encoded)
            if manifest["format"] != "fxopt-dataset-v1":
                raise ValueError("unsupported format")
            if any(not isinstance(manifest[key], str) or not manifest[key].strip()
                   for key in ("dataset_id", "revision")):
                raise ValueError("missing dataset identity")
            entry = manifest["files"][path.name]
            with path.open("rb") as handle:
                size = path.stat().st_size
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if digest != entry["sha256"] or size != entry["bytes"]:
                raise ValueError("input bytes differ from the published manifest")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ConfigError(f"invalid published dataset input {path}: {exc}") from exc
        result[name] = {
            "dataset_id": manifest["dataset_id"], "revision": manifest["revision"],
            "file": path.name, "sha256": digest, "bytes": size,
            "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
        }
    return result
