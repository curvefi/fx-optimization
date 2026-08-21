"""Frozen pair specification contract and loader."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .common import (
    SpecError,
    repository_relative,
)
from .registry import SpecRegistry


@dataclass(frozen=True)
class PairSpec:
    """Immutable pair specification."""

    id: str
    name: str
    base_token: str
    quote_token: str
    base_decimals: int = 18
    quote_decimals: int = 18
    tags: tuple[str, ...] = ()
    source_path: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise SpecError("pair id must be a non-empty string")
        if not isinstance(self.base_token, str) or not self.base_token:
            raise SpecError("pair base_token is required and must be a non-empty string")
        if not isinstance(self.quote_token, str) or not self.quote_token:
            raise SpecError("pair quote_token is required and must be a non-empty string")
        if self.base_decimals <= 0 or self.quote_decimals <= 0:
            raise SpecError("token decimals must be positive integers")

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "base_token": self.base_token,
            "quote_token": self.quote_token,
            "base_decimals": self.base_decimals,
            "quote_decimals": self.quote_decimals,
            "tags": list(self.tags),
            "source_path": self.source_path.as_posix() if self.source_path else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PairSpec:
        """Reconstruct the non-economic pair identity contract."""
        unknown = sorted(
            set(data)
            - {
                "id",
                "name",
                "base_token",
                "quote_token",
                "base_decimals",
                "quote_decimals",
                "tags",
                "source_path",
            }
        )
        if unknown:
            raise SpecError("unsupported pair fields: " + ", ".join(unknown))
        return cls(
            id=str(data["id"]),
            name=str(data.get("name", data["id"])),
            base_token=data["base_token"],
            quote_token=data["quote_token"],
            base_decimals=int(data.get("base_decimals", 18)),
            quote_decimals=int(data.get("quote_decimals", 18)),
            tags=tuple(str(t) for t in data.get("tags", ())),
            source_path=Path(data["source_path"]) if data.get("source_path") else None,
        )


def load_pair_spec(
    path_or_id: str | os.PathLike[str],
    *,
    repository: Path,
) -> PairSpec:
    """Load and validate a pair TOML file or search by id/alias."""
    registry = SpecRegistry.from_root(repository)
    root = registry.context.project_root
    candidate = registry.resolve("pair", path_or_id)

    with candidate.open("rb") as stream:
        raw_data = tomllib.load(stream)

    pair_data = raw_data.get("pair", raw_data)
    known = {
        "id",
        "name",
        "base_token",
        "quote_token",
        "base_decimals",
        "quote_decimals",
        "tags",
    }
    unknown = sorted(set(pair_data) - known)
    if unknown:
        raise SpecError("unsupported pair fields: " + ", ".join(unknown))

    pair_id = pair_data.get("id") or candidate.stem
    name = pair_data.get("name") or pair_id
    base_token = pair_data.get("base_token")
    quote_token = pair_data.get("quote_token")
    base_decimals = int(pair_data.get("base_decimals", 18))
    quote_decimals = int(pair_data.get("quote_decimals", 18))

    tags = tuple(pair_data.get("tags", []))

    source_path = repository_relative(candidate, root)

    return PairSpec(
        id=pair_id,
        name=name,
        base_token=base_token,
        quote_token=quote_token,
        base_decimals=base_decimals,
        quote_decimals=quote_decimals,
        tags=tags,
        source_path=source_path,
    )


__all__ = ["PairSpec", "load_pair_spec"]
