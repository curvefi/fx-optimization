"""Frozen shiftclick diagnostic specification contract and loader."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .common import (
    SpecError,
    assert_contained_path,
    repository_relative,
    repository_root,
    serializable,
)


@dataclass(frozen=True)
class ShiftclickSpec:
    """Immutable shiftclick detailed trace / economic diagnostic contract."""

    id: str
    source_kind: str
    source_run_id: str
    selection_kind: str
    selection_value: Any = None
    pair_id: str = ""
    scenario_id: str = ""
    policy_id: str = ""
    trace_interval: int = 1
    trace_actions: bool = True
    tags: tuple[str, ...] = ()
    source_spec_path: Path | None = None

    def __post_init__(self) -> None:
        if not self.id or not self.id.strip():
            raise SpecError("shiftclick id must be non-empty")
        if not self.source_run_id:
            raise SpecError("shiftclick source_run_id must be non-empty")
        if not self.pair_id or not self.scenario_id or not self.policy_id:
            raise SpecError("shiftclick requires pair_id, scenario_id, and policy_id")
        if isinstance(self.trace_interval, bool) or not isinstance(self.trace_interval, int):
            raise SpecError("trace_interval must be an integer")
        if self.trace_interval <= 0:
            raise SpecError("trace_interval must be a positive integer")
        if not isinstance(self.trace_actions, bool):
            raise SpecError("trace_actions must be a boolean")
        if self.source_kind not in {"grid", "optimization"}:
            raise SpecError(f"unsupported shiftclick source_kind: {self.source_kind!r}")
        allowed = {"coordinates", "candidate_id", "index"} if self.source_kind == "grid" else {"best"}
        if self.selection_kind not in allowed:
            raise SpecError(f"unsupported shiftclick selection_kind: {self.selection_kind!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_kind": self.source_kind,
            "source_run_id": self.source_run_id,
            "selection_kind": self.selection_kind,
            "selection_value": serializable(self.selection_value),
            "pair_id": self.pair_id,
            "scenario_id": self.scenario_id,
            "policy_id": self.policy_id,
            "trace_interval": self.trace_interval,
            "trace_actions": self.trace_actions,
            "tags": list(self.tags),
            "source_spec_path": self.source_spec_path.as_posix() if self.source_spec_path else None,
        }


def load_shiftclick_spec(
    path_or_id: str | os.PathLike[str],
    *,
    repository: Path | None = None,
) -> ShiftclickSpec:
    """Load and validate a shiftclick TOML specification."""
    root = repository.resolve() if repository is not None else repository_root()
    candidate = Path(path_or_id)

    if not candidate.is_file():
        search_paths = [
            root / "shiftclick" / "specs" / f"{path_or_id}.toml",
            root / "configs" / "shiftclick" / f"{path_or_id}.toml",
            root / "configs" / f"{path_or_id}.toml",
        ]
        found = None
        for p in search_paths:
            if p.is_file():
                found = p
                break
        if found is None:
            raise FileNotFoundError(f"Shiftclick specification not found: {path_or_id}")
        candidate = found

    assert_contained_path(candidate, root, allow_symlinks=True)

    with candidate.open("rb") as stream:
        raw_data = tomllib.load(stream)

    sc_data = raw_data.get("shiftclick", raw_data)
    known = {
        "id",
        "source_kind",
        "source_run_id",
        "selection_kind",
        "selection_value",
        "pair_id",
        "scenario_id",
        "policy_id",
        "trace_interval",
        "trace_actions",
        "tags",
    }
    unknown = sorted(set(sc_data) - known)
    if unknown:
        raise SpecError("unsupported shiftclick fields: " + ", ".join(unknown))

    sc_id = sc_data.get("id") or candidate.stem

    tags = tuple(sc_data.get("tags", []))
    source_spec_path = repository_relative(candidate, root)

    return ShiftclickSpec(
        id=sc_id,
        source_kind=str(sc_data.get("source_kind", "")),
        source_run_id=str(sc_data.get("source_run_id", "")),
        selection_kind=str(sc_data.get("selection_kind", "")),
        selection_value=sc_data.get("selection_value"),
        pair_id=str(sc_data.get("pair_id", "")),
        scenario_id=str(sc_data.get("scenario_id", "")),
        policy_id=str(sc_data.get("policy_id", "")),
        trace_interval=sc_data.get("trace_interval", 1),
        trace_actions=sc_data.get("trace_actions", True),
        tags=tags,
        source_spec_path=source_spec_path,
    )


__all__ = ["ShiftclickSpec", "load_shiftclick_spec"]
