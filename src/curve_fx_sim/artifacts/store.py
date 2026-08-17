"""Repository-relative run storage, workspace management, and artifact access."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from ..specs.common import assert_contained_path, repository_root
from .manifest import load_manifest, write_manifest_atomic
from .tables import EvaluationTable


class RunStore:
    """Manages repository-relative immutable runs under `<repo_root>/runs/`."""

    def __init__(self, root: str | os.PathLike[str] | None = None) -> None:
        self.root_dir = repository_root(root).resolve()
        self.runs_dir = (self.root_dir / "runs").resolve()
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def allocate_run_dir(self, run_kind: str, run_id: str) -> Path:
        """Allocate a new immutable run directory strictly contained in runs/."""
        if any(c in run_id for c in "/\\") or run_id in {".", "..", "latest"}:
            raise ValueError(f"invalid run_id: {run_id!r}")
        run_path = self.runs_dir / run_id
        assert_contained_path(run_path, self.runs_dir, allow_symlinks=False)
        if run_path.exists():
            raise FileExistsError(f"immutable run directory already exists: {run_id}")
        run_path.mkdir(parents=True, exist_ok=False)
        return run_path

    def get_run_dir(self, run_id: str) -> Path:
        """Return the directory for an existing run_id."""
        if any(c in run_id for c in "/\\") or run_id in {".", "..", "latest"}:
            raise ValueError(f"invalid run_id: {run_id!r}")
        run_path = self.runs_dir / run_id
        assert_contained_path(run_path, self.runs_dir, allow_symlinks=False)
        if not run_path.is_dir():
            raise FileNotFoundError(f"run directory not found: {run_path}")
        return run_path

    def save_manifest(
        self,
        run_id: str,
        manifest: Mapping[str, Any],
        *,
        expected_kind: str | None = None,
    ) -> Path:
        """Atomically validate and save a run manifest inside its run directory."""
        run_path = self.runs_dir / run_id
        run_dir = self.allocate_run_dir(manifest.get("run_kind", "grid"), run_id) if not run_path.exists() else self.get_run_dir(run_id)
        return write_manifest_atomic(run_dir / "manifest.json", manifest, expected_kind=expected_kind)

    def load_manifest(
        self,
        run_id_or_path: str | os.PathLike[str],
        expected_kind: str | None = None,
    ) -> dict[str, Any]:
        """Load a manifest from a run_id or explicit path."""
        candidate = Path(run_id_or_path)
        if candidate.is_dir():
            candidate = candidate / "manifest.json"
        if candidate.is_file():
            assert_contained_path(candidate, self.root_dir, allow_symlinks=True)
            return load_manifest(candidate, expected_kind=expected_kind)
        return load_manifest(self.get_run_dir(str(run_id_or_path)) / "manifest.json", expected_kind=expected_kind)

    def save_evaluation_table(self, run_id: str, table: EvaluationTable) -> Path:
        """Atomically save the run's sole compact evaluation table."""
        return table.to_npz(self.get_run_dir(run_id) / "evaluation_table.npz")

__all__ = ["RunStore"]
