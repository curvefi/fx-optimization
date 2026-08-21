"""Lexically safe run-scoped paths for shared-NFS execution."""

from __future__ import annotations

import re
from pathlib import PurePosixPath


class RunPathError(ValueError):
    """Raised when a run identifier can escape its namespace."""


_RUN_ID = re.compile(r"^[A-Za-z0-9._-]+$")


def validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not run_id or ".." in run_id or not _RUN_ID.fullmatch(run_id):
        raise RunPathError(f"invalid run_id {run_id!r}")
    return run_id


def remote_run_paths(
    run_id: str,
    remote_base: PurePosixPath | str = "/home/heswithme/arb",
) -> dict[str, PurePosixPath]:
    root = PurePosixPath(str(remote_base)) / "runs" / validate_run_id(run_id)
    return {
        "root": root,
        "inputs": root / "inputs",
        "shards": root / "shards",
        "results": root / "results",
        "logs": root / "logs",
        "data": root / "data",
        "manifest": root / "manifest.json",
    }


__all__ = ["RunPathError", "remote_run_paths", "validate_run_id"]
