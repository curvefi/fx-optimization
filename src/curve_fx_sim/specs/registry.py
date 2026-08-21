"""Canonical project-contained resolution for checked-in specifications."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .common import ProjectContext, assert_contained_path


_CONFIG_DIRS = {
    "pair": "pairs",
    "scenario": "scenarios",
    "policy": "policies",
    "grid": "grids",
    "optimization": "optimization",
    "site": "sites",
}


@dataclass(frozen=True)
class SpecRegistry:
    """Resolve spec IDs in canonical directories and explicit paths from the project root."""

    context: ProjectContext

    @classmethod
    def from_root(
        cls, repository: str | os.PathLike[str]
    ) -> SpecRegistry:
        return cls(ProjectContext.from_root(repository))

    def resolve(
        self,
        kind: str,
        path_or_id: str | os.PathLike[str],
        *,
        explicit_only: bool = False,
    ) -> Path:
        spelling = os.fspath(path_or_id)
        raw = Path(path_or_id)
        root = self.context.project_root
        is_bare = not raw.is_absolute() and raw.parent == Path(".") and not spelling.startswith("./")
        if explicit_only and is_bare:
            raise ValueError(f"explicit specification path required, got bare name: {path_or_id}")
        is_identifier = is_bare

        if is_identifier:
            try:
                config_dir = _CONFIG_DIRS[kind]
            except KeyError as exc:
                raise ValueError(f"unsupported specification kind: {kind}") from exc
            filename = raw.name if raw.suffix == ".toml" else f"{raw.name}.toml"
            candidate = self.context.config_root / config_dir / filename
        else:
            candidate = raw if raw.is_absolute() else root / raw

        resolved = assert_contained_path(candidate, root, allow_symlinks=True)
        if not resolved.is_file():
            label = "path" if not is_identifier else f"{kind} specification"
            raise FileNotFoundError(f"{label} not found: {path_or_id} ({resolved})")
        return resolved


__all__ = ["SpecRegistry"]
