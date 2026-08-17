"""Safe run-scoped staging and bundle hashing for cluster execution.

All remote cluster artifacts (sources, inputs, shard ranges, build products,
results, and logs) are strictly confined below an immutable run namespace
``<remote_base>/runs/<run_id>/``. No paths escape or use relative '..' traversal.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Final, Mapping

from ..artifacts.io import atomic_write_json
from ..specs.common import assert_contained_path, repository_root
from ..specs.scenario import ScenarioSpec

_RUN_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9._-]+$")


class StagingError(ValueError):
    """Raised when staging paths escape containment or run IDs are invalid."""


def _containment_error(value: object, root: PurePosixPath, resolved: PurePosixPath | None = None) -> StagingError:
    detail = f" resolved to {resolved}" if resolved is not None else ""
    return StagingError(f"path containment violation: {value!r} escapes run root {root}{detail}")


def validate_run_id(run_id: str) -> str:
    """Validate that run_id is a non-empty, single-component POSIX-safe identifier."""
    if not run_id or not isinstance(run_id, str):
        raise StagingError(f"run_id must be a non-empty string, got {run_id!r}")
    if ".." in run_id or "/" in run_id or "\\" in run_id:
        raise StagingError(f"path containment violation: run_id {run_id!r} escapes its run namespace")
    if not _RUN_ID_PATTERN.match(run_id):
        raise StagingError(
            f"invalid run_id {run_id!r}: must match regex '^[A-Za-z0-9._-]+$' and contain no slashes or '..'"
        )
    return run_id


def remote_run_paths(
    run_id: str,
    remote_base: PurePosixPath | str = "/home/heswithme/arb",
) -> dict[str, PurePosixPath]:
    """Return canonical paths strictly scoped below ``<remote_base>/runs/<run_id>``."""
    clean_id = validate_run_id(run_id)
    base = PurePosixPath(str(remote_base))
    run_root = base / "runs" / clean_id

    return {
        "root": run_root,
        "src": run_root / "src",
        "inputs": run_root / "inputs",
        "shards": run_root / "shards",
        "results": run_root / "results",
        "logs": run_root / "logs",
        "build": run_root / "build",
        "data": run_root / "data",
        "manifest": run_root / "manifest.json",
    }


def scoped_remote_path(
    run_id: str,
    subpath: str | Path | PurePosixPath,
    remote_base: PurePosixPath | str = "/home/heswithme/arb",
) -> PurePosixPath:
    """Safely resolve a subpath strictly within the remote run root."""
    paths = remote_run_paths(run_id, remote_base=remote_base)
    root = paths["root"]
    raw = str(subpath).strip()

    if raw.startswith("/"):
        candidate = PurePosixPath(raw)
        try:
            candidate.relative_to(root)
        except ValueError:
            raise _containment_error(raw, root) from None
        return candidate

    candidate = root / raw
    parts: list[str] = []
    for part in candidate.parts:
        if part == "..":
            if parts and parts[-1] != "":
                parts.pop()
        elif part != ".":
            parts.append(part)

    normalized = PurePosixPath(*parts)
    try:
        normalized.relative_to(root)
    except ValueError:
        raise _containment_error(subpath, root, normalized) from None
    return normalized


def sha256_path(path: Path) -> str:
    """Compute SHA-256 digest of a local file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class MarketFileEntry:
    """A referenced market input file with exact local/remote paths and checksum."""

    local_path: Path
    remote_path: PurePosixPath
    sha256: str
    kind: str = "market"

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.remote_path),
            "local_path": str(self.local_path.as_posix()),
            "sha256": self.sha256,
            "kind": self.kind,
        }


@dataclass(frozen=True)
class WorkBundle:
    """An immutable, fully hashed execution bundle for local and remote blades."""

    run_id: str
    manifest_local: Path
    manifest_remote: PurePosixPath
    manifest_sha256: str
    session_manifest_local: Path
    session_manifest_remote: PurePosixPath
    session_manifest_local_sha256: str
    session_manifest_remote_sha256: str
    session_config: dict[str, Any] = field(default_factory=dict)
    template_local: Path | None = None
    template_remote: PurePosixPath | None = None
    template_sha256: str | None = None
    market_files: tuple[MarketFileEntry, ...] = field(default_factory=tuple)
    shards_local_dir: Path | None = None
    shards_remote_dir: PurePosixPath | None = None

def prepare_work_bundle(
    manifest_file: Path,
    root: Path | None = None,
    remote_base: PurePosixPath | str = "/home/heswithme/arb",
) -> WorkBundle:
    """Inspect a run directory and manifest to construct a complete WorkBundle."""
    manifest_resolved = manifest_file.resolve()
    with manifest_resolved.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    run_id = validate_run_id(str(manifest.get("run_id", manifest_resolved.parent.name)))
    paths = remote_run_paths(run_id, remote_base=remote_base)

    repo_root = repository_root(root)
    manifest_sha = sha256_path(manifest_resolved)

    # Inputs are owned by the immutable resolved scenario contract.
    resolved_spec = manifest.get("resolved_spec")
    scenario = resolved_spec.get("scenario") if isinstance(resolved_spec, Mapping) else None
    if not isinstance(scenario, Mapping):
        raise StagingError("manifest resolved_spec.scenario must be an object")
    scenario_spec = ScenarioSpec.from_dict(scenario)
    session_config = scenario_spec.harness_session_config()

    template_local: Path | None = None
    template_remote: PurePosixPath | None = None
    template_sha: str | None = None

    tpl_val = scenario.get("template_path")
    if not tpl_val:
        raise StagingError("manifest resolved_spec.scenario.template_path must be non-empty")
    tpl_rel = str(tpl_val)
    tpl_cand = repo_root / tpl_rel
    assert_contained_path(tpl_cand, repo_root, allow_symlinks=True)
    if not tpl_cand.is_file():
        raise StagingError(f"scenario template does not exist: {tpl_rel}")
    template_local = tpl_cand.resolve()
    template_remote = scoped_remote_path(run_id, tpl_rel, remote_base=remote_base)
    template_sha = sha256_path(template_local)
    declared_template_sha = scenario.get("template_sha256")
    if declared_template_sha and str(declared_template_sha).lower() != template_sha.lower():
        raise StagingError(
            "scenario template SHA-256 mismatch: "
            f"expected {declared_template_sha}, calculated {template_sha}"
        )

    market_entries: list[MarketFileEntry] = []
    market_refs = scenario.get("market_files") or []

    for item in market_refs:
        rel_str = item if isinstance(item, str) else item.get("path")
        if not rel_str:
            raise StagingError("scenario market_files entry path must be non-empty")
        rel_str = str(rel_str)
        local_path = repo_root / rel_str
        assert_contained_path(local_path, repo_root, allow_symlinks=True)
        if not local_path.is_file():
            raise StagingError(f"scenario market file does not exist: {rel_str}")
        calculated_sha256 = sha256_path(local_path)
        expected_sha256 = item.get("sha256") if isinstance(item, Mapping) else None
        if expected_sha256 and str(expected_sha256).lower() != calculated_sha256.lower():
            raise StagingError(
                f"scenario market file SHA-256 mismatch for {rel_str}: "
                f"expected {expected_sha256}, calculated {calculated_sha256}"
            )
        market_entries.append(
            MarketFileEntry(
                local_path=local_path.resolve(),
                remote_path=scoped_remote_path(run_id, rel_str, remote_base=remote_base),
                sha256=calculated_sha256,
                kind=str(item.get("kind", "market")) if isinstance(item, Mapping) else "market",
            )
        )

    inputs_dir = manifest_resolved.parent / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    local_session_manifest = inputs_dir / "session_manifest.local.json"
    remote_session_manifest_local = inputs_dir / "session_manifest.remote.json"
    remote_session_manifest = paths["inputs"] / "session_manifest.json"

    def materialized_manifest(*, remote: bool) -> dict[str, Any]:
        market_files = [
            {
                "path": str(entry.remote_path if remote else entry.local_path),
                "sha256": entry.sha256,
                "kind": entry.kind,
            }
            for entry in market_entries
        ]
        return {
            "schema_version": "fxsim_manifest_v1",
            "run_kind": "session",
            "run_id": run_id,
            "resolved_spec": {
                "scenario": scenario_spec.harness_manifest_scenario(market_files)
            },
        }

    atomic_write_json(local_session_manifest, materialized_manifest(remote=False))
    atomic_write_json(remote_session_manifest_local, materialized_manifest(remote=True))

    shards_dir = manifest_resolved.parent / "shards"
    shards_local = shards_dir if shards_dir.is_dir() else None
    shards_remote = paths["shards"]

    return WorkBundle(
        run_id=run_id,
        manifest_local=manifest_resolved,
        manifest_remote=paths["manifest"],
        manifest_sha256=manifest_sha,
        session_manifest_local=local_session_manifest,
        session_manifest_remote=remote_session_manifest,
        session_manifest_local_sha256=sha256_path(local_session_manifest),
        session_manifest_remote_sha256=sha256_path(remote_session_manifest_local),
        session_config=session_config,
        template_local=template_local,
        template_remote=template_remote,
        template_sha256=template_sha,
        market_files=tuple(market_entries),
        shards_local_dir=shards_local,
        shards_remote_dir=shards_remote,
    )


__all__ = [
    "MarketFileEntry",
    "StagingError",
    "WorkBundle",
    "prepare_work_bundle",
    "remote_run_paths",
    "scoped_remote_path",
    "sha256_path",
    "validate_run_id",
]
