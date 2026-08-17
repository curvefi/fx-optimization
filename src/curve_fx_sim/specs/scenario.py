"""Frozen scenario specification contract and loader."""

from __future__ import annotations

import hashlib
import math
import os
import re
import tomllib
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import (
    SpecError,
    assert_contained_path,
    canonical_json_bytes,
    repository_relative,
    repository_root,
)


def _strict_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise SpecError(f"{label} must be an integer, got {value!r}")
    numeric = float(value)
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise SpecError(f"{label} must be an integer, got {value!r}")
    return int(value)


def _strict_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise SpecError(f"{label} must be numeric, got {value!r}")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise SpecError(f"{label} must be finite, got {value!r}")
    return numeric


def _strict_bool(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise SpecError(f"{label} must be a boolean, got {value!r}")
    return value


def _validate_sha256(value: str | None, *, label: str) -> None:
    if value is not None and not re.fullmatch(r"[0-9a-f]{64}", value):
        raise SpecError(f"{label} must be a lowercase 64-character SHA-256")


@dataclass(frozen=True)
class MarketFileRef:
    """Attested reference to a market or trade event data file."""

    path: Path
    sha256: str | None = None
    kind: str = "market"  # market or chainlink

    def __post_init__(self) -> None:
        if not self.path.as_posix().strip():
            raise SpecError("market file path must be non-empty")
        if self.kind not in {"market", "chainlink"}:
            raise SpecError(
                f"market file {self.path} has unsupported kind {self.kind!r}"
            )
        _validate_sha256(self.sha256, label=f"market file {self.path} sha256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path.as_posix(),
            "sha256": self.sha256,
            "kind": self.kind,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MarketFileRef:
        unknown = sorted(set(data) - {"path", "sha256", "kind"})
        if unknown:
            raise SpecError("unsupported market file fields: " + ", ".join(unknown))
        return cls(
            path=Path(data["path"]),
            sha256=data.get("sha256"),
            kind=str(data.get("kind", "market")),
        )


@dataclass(frozen=True)
class ScenarioSpec:
    """Immutable scenario specification."""

    id: str
    pair_id: str
    name: str
    description: str = ""
    start_time: int = 0
    end_time: int = 0
    n_candles: int = 0
    candle_filter: float = 99.0
    min_swap: float = 1e-6
    max_swap: float = 1.0
    dustswap_freq_s: int = 3600
    dustswap_random: bool = False
    dustswap_dynamic_freq_s: int = 0
    dustswap_dynamic_gap_enabled: bool = False
    dustswap_dynamic_gap_bps: float = 0.0
    dustswap_dynamic_heartbeat_s: int = 0
    dustswap_commit_clock_freq_s: int = 0
    policy_keeper_enabled: bool = False
    allow_hybrid_keeper: bool = False
    user_swap_freq_s: int = 0
    user_swap_size_frac: float = 0.01
    user_swap_thresh: float = 0.05
    disable_slippage_probes: bool = False
    yb_mode: str = "off"  # YieldBasis mode: off | passive | active_2l
    yb_releverage: bool = False  # legacy alias; reconciled in __post_init__
    yb_releverage_fee: float = 0.012
    yb_cash_multiplier: float = 1.0
    market_files: tuple[MarketFileRef, ...] = ()
    template_path: Path | None = None
    template_sha256: str | None = None
    tags: tuple[str, ...] = ()
    source_path: Path | None = None

    def __post_init__(self) -> None:
        if not self.id or not self.id.strip():
            raise SpecError("scenario id must be non-empty")
        if not self.pair_id or not self.pair_id.strip():
            raise SpecError("scenario pair_id must be non-empty")
        for name in (
            "start_time",
            "end_time",
            "n_candles",
            "dustswap_freq_s",
            "dustswap_dynamic_freq_s",
            "dustswap_dynamic_heartbeat_s",
            "dustswap_commit_clock_freq_s",
            "user_swap_freq_s",
        ):
            if _strict_int(getattr(self, name), label=name) < 0:
                raise SpecError(f"{name} must be non-negative")
        if self.start_time and self.end_time and self.end_time <= self.start_time:
            raise SpecError("end_time must be greater than start_time")
        if not 0 < _strict_float(self.candle_filter, label="candle_filter") <= 100:
            raise SpecError("candle_filter must be in (0, 100]")
        min_swap = _strict_float(self.min_swap, label="min_swap")
        max_swap = _strict_float(self.max_swap, label="max_swap")
        if min_swap <= 0 or max_swap < min_swap:
            raise SpecError("swap bounds require 0 < min_swap <= max_swap")
        for name in ("dustswap_dynamic_gap_bps", "user_swap_thresh"):
            if _strict_float(getattr(self, name), label=name) < 0:
                raise SpecError(f"{name} must be non-negative")
        user_swap_size = _strict_float(self.user_swap_size_frac, label="user_swap_size_frac")
        if not 0 < user_swap_size <= 1:
            raise SpecError("user_swap_size_frac must be in (0, 1]")
        yb_fee = _strict_float(self.yb_releverage_fee, label="yb_releverage_fee")
        if not 0 <= yb_fee <= 1:
            raise SpecError("yb_releverage_fee must be in [0, 1]")
        if self.yb_mode not in {"off", "passive", "active_2l"}:
            raise SpecError(
                "yb_mode must be one of 'off', 'passive', 'active_2l'"
            )
        # Reconcile the legacy boolean alias with the canonical mode:
        # yb_releverage=true maps to the active 2L model, and any active mode
        # (passive or active_2l) keeps the legacy flag true.
        if self.yb_mode == "off" and self.yb_releverage:
            object.__setattr__(self, "yb_mode", "active_2l")
        elif self.yb_mode != "off" and not self.yb_releverage:
            object.__setattr__(self, "yb_releverage", True)
        if _strict_float(
            self.yb_cash_multiplier,
            label="yb_cash_multiplier",
        ) <= 0:
            raise SpecError("yb_cash_multiplier must be positive")
        _validate_sha256(self.template_sha256, label="template_sha256")
        paths = [item.path.as_posix() for item in self.market_files]
        if len(paths) != len(set(paths)):
            raise SpecError("scenario market_files contains duplicate paths")
        market_count = sum(item.kind == "market" for item in self.market_files)
        chainlink_count = sum(item.kind == "chainlink" for item in self.market_files)
        if market_count > 1 or chainlink_count > 1:
            raise SpecError("scenario supports one market file and at most one Chainlink file")

    def harness_session_config(self) -> dict[str, Any]:
        """Return exactly the typed settings consumed by ``open_session``."""
        return {
            "n_candles": self.n_candles,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "candle_filter": self.candle_filter,
            "min_swap": self.min_swap,
            "max_swap": self.max_swap,
            "dustswap_freq_s": self.dustswap_freq_s,
            "dustswap_random": self.dustswap_random,
            "dustswap_dynamic_freq_s": self.dustswap_dynamic_freq_s,
            "dustswap_dynamic_gap_enabled": self.dustswap_dynamic_gap_enabled,
            "dustswap_dynamic_gap_bps": self.dustswap_dynamic_gap_bps,
            "dustswap_dynamic_heartbeat_s": self.dustswap_dynamic_heartbeat_s,
            "dustswap_commit_clock_freq_s": self.dustswap_commit_clock_freq_s,
            "policy_keeper_enabled": self.policy_keeper_enabled,
            "allow_hybrid_keeper": self.allow_hybrid_keeper,
            "user_swap_freq_s": self.user_swap_freq_s,
            "user_swap_size_frac": self.user_swap_size_frac,
            "user_swap_thresh": self.user_swap_thresh,
            "disable_slippage_probes": self.disable_slippage_probes,
            "yb_mode": self.yb_mode,
            "yb_releverage": self.yb_releverage,
            "yb_releverage_fee": self.yb_releverage_fee,
            "yb_cash_multiplier": self.yb_cash_multiplier,
        }

    def harness_manifest_scenario(
        self,
        market_files: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Return the evaluator's sole narrow market-manifest contract."""
        files = market_files if market_files is not None else [
            market_file.to_dict() for market_file in self.market_files
        ]
        return {
            "id": self.id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "n_candles": self.n_candles,
            "candle_filter": self.candle_filter,
            "market_files": [
                {
                    "path": str(item["path"]),
                    "kind": str(item.get("kind", "market")),
                    "sha256": item.get("sha256"),
                }
                for item in files
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "id": self.id,
            "pair_id": self.pair_id,
            "name": self.name,
            "description": self.description,
            **self.harness_session_config(),
            "market_files": [f.to_dict() for f in self.market_files],
            "template_path": self.template_path.as_posix() if self.template_path else None,
            "template_sha256": self.template_sha256,
            "tags": list(self.tags),
            "source_path": self.source_path.as_posix() if self.source_path else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ScenarioSpec:
        """Reconstruct ScenarioSpec preserving all market files, template paths, and metadata."""
        known = {
            "id",
            "pair_id",
            "name",
            "description",
            "start_time",
            "end_time",
            "n_candles",
            "candle_filter",
            "min_swap",
            "max_swap",
            "dustswap_freq_s",
            "dustswap_random",
            "dustswap_dynamic_freq_s",
            "dustswap_dynamic_gap_enabled",
            "dustswap_dynamic_gap_bps",
            "dustswap_dynamic_heartbeat_s",
            "dustswap_commit_clock_freq_s",
            "policy_keeper_enabled",
            "allow_hybrid_keeper",
            "user_swap_freq_s",
            "user_swap_size_frac",
            "user_swap_thresh",
            "disable_slippage_probes",
            "yb_mode",
            "yb_releverage",
            "yb_releverage_fee",
            "yb_cash_multiplier",
            "market_files",
            "template_path",
            "template_sha256",
            "tags",
            "source_path",
        }
        unknown = sorted(set(data) - known)
        if unknown:
            raise SpecError("unsupported scenario fields: " + ", ".join(unknown))
        m_files = tuple(
            MarketFileRef.from_dict(f) if isinstance(f, Mapping) else MarketFileRef(path=Path(f))
            for f in data.get("market_files", ())
        )
        return cls(
            id=str(data["id"]),
            pair_id=str(data["pair_id"]),
            name=str(data.get("name", data["id"])),
            description=str(data.get("description", "")),
            start_time=_strict_int(data.get("start_time", 0), label="start_time"),
            end_time=_strict_int(data.get("end_time", 0), label="end_time"),
            n_candles=_strict_int(data.get("n_candles", 0), label="n_candles"),
            candle_filter=_strict_float(data.get("candle_filter", 99.0), label="candle_filter"),
            min_swap=_strict_float(data.get("min_swap", 1e-6), label="min_swap"),
            max_swap=_strict_float(data.get("max_swap", 1.0), label="max_swap"),
            dustswap_freq_s=_strict_int(data.get("dustswap_freq_s", 3600), label="dustswap_freq_s"),
            dustswap_random=_strict_bool(data.get("dustswap_random", False), label="dustswap_random"),
            dustswap_dynamic_freq_s=_strict_int(data.get("dustswap_dynamic_freq_s", 0), label="dustswap_dynamic_freq_s"),
            dustswap_dynamic_gap_enabled=_strict_bool(data.get("dustswap_dynamic_gap_enabled", False), label="dustswap_dynamic_gap_enabled"),
            dustswap_dynamic_gap_bps=_strict_float(data.get("dustswap_dynamic_gap_bps", 0.0), label="dustswap_dynamic_gap_bps"),
            dustswap_dynamic_heartbeat_s=_strict_int(data.get("dustswap_dynamic_heartbeat_s", 0), label="dustswap_dynamic_heartbeat_s"),
            dustswap_commit_clock_freq_s=_strict_int(data.get("dustswap_commit_clock_freq_s", 0), label="dustswap_commit_clock_freq_s"),
            policy_keeper_enabled=_strict_bool(data.get("policy_keeper_enabled", False), label="policy_keeper_enabled"),
            allow_hybrid_keeper=_strict_bool(data.get("allow_hybrid_keeper", False), label="allow_hybrid_keeper"),
            user_swap_freq_s=_strict_int(data.get("user_swap_freq_s", 0), label="user_swap_freq_s"),
            user_swap_size_frac=_strict_float(data.get("user_swap_size_frac", 0.01), label="user_swap_size_frac"),
            user_swap_thresh=_strict_float(data.get("user_swap_thresh", 0.05), label="user_swap_thresh"),
            disable_slippage_probes=_strict_bool(data.get("disable_slippage_probes", False), label="disable_slippage_probes"),
            yb_mode=str(data.get("yb_mode", "off")),
            yb_releverage=_strict_bool(data.get("yb_releverage", False), label="yb_releverage"),
            yb_releverage_fee=_strict_float(data.get("yb_releverage_fee", 0.012), label="yb_releverage_fee"),
            yb_cash_multiplier=_strict_float(data.get("yb_cash_multiplier", 1.0), label="yb_cash_multiplier"),
            market_files=m_files,
            template_path=Path(data["template_path"]) if data.get("template_path") else None,
            template_sha256=str(data["template_sha256"]) if data.get("template_sha256") else None,
            tags=tuple(str(t) for t in data.get("tags", ())),
            source_path=Path(data["source_path"]) if data.get("source_path") else None,
        )

    def scenario_fingerprint(self) -> str:
        """Compute deterministic SHA-256 fingerprint of the scenario inputs."""
        payload = {
            "id": self.id,
            "pair_id": self.pair_id,
            **self.harness_session_config(),
            "market_files": [f.to_dict() for f in self.market_files],
            "template_sha256": self.template_sha256,
        }
        return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def load_scenario_spec(
    path_or_id: str | os.PathLike[str],
    *,
    repository: Path | None = None,
) -> ScenarioSpec:
    """Load and validate a scenario TOML specification."""
    root = repository.resolve() if repository is not None else repository_root()
    candidate = Path(path_or_id)

    if not candidate.is_file():
        search_paths = [
            root / "configs" / "scenarios" / f"{path_or_id}.toml",
            root / "configs" / f"{path_or_id}.toml",
            root / "data" / "scenarios" / f"{path_or_id}.toml",
        ]
        found = None
        for p in search_paths:
            if p.is_file():
                found = p
                break
        if found is None:
            raise FileNotFoundError(f"Scenario specification not found: {path_or_id}")
        candidate = found

    assert_contained_path(candidate, root, allow_symlinks=True)

    with candidate.open("rb") as stream:
        raw_data = tomllib.load(stream)

    scenario_data = raw_data.get("scenario", raw_data)

    known_source_fields = {
        "id",
        "pair_id",
        "pair",
        "name",
        "description",
        "start_time",
        "end_time",
        "n_candles",
        "candle_filter",
        "min_swap",
        "max_swap",
        "dustswap_freq_s",
        "dustswap_random",
        "dustswap_dynamic_freq_s",
        "dustswap_dynamic_gap_enabled",
        "dustswap_dynamic_gap_bps",
        "dustswap_dynamic_heartbeat_s",
        "dustswap_commit_clock_freq_s",
        "policy_keeper_enabled",
        "allow_hybrid_keeper",
        "user_swap_freq_s",
        "user_swap_size_frac",
        "user_swap_thresh",
        "disable_slippage_probes",
        "yb_mode",
        "yb_releverage",
        "yb_releverage_fee",
        "yb_cash_multiplier",
        "market_files",
        "template_path",
        "template",
        "template_sha256",
        "tags",
    }
    unknown = sorted(set(scenario_data) - known_source_fields)
    if unknown:
        raise SpecError("unsupported scenario fields: " + ", ".join(unknown))

    scenario_id = scenario_data.get("id") or candidate.stem
    pair_id = scenario_data.get("pair_id") or scenario_data.get("pair", "")
    market_files_raw = scenario_data.get("market_files", [])
    market_files: list[dict[str, Any]] = []
    for item in market_files_raw:
        if isinstance(item, str):
            p = Path(item)
            resolved = repository_relative(root / p if not p.is_absolute() else p, root)
            market_files.append({"path": resolved.as_posix(), "kind": "market"})
        elif isinstance(item, Mapping):
            p = Path(item["path"])
            resolved = repository_relative(root / p if not p.is_absolute() else p, root)
            market_files.append(
                {
                    "path": resolved.as_posix(),
                    "sha256": item.get("sha256"),
                    "kind": item.get("kind", "market"),
                }
            )
        else:
            raise SpecError("scenario market_files entries must be strings or tables")
    if not market_files:
        raise SpecError("scenario must declare a non-empty market_files list")

    template_raw = scenario_data.get("template_path") or scenario_data.get("template")
    template_path: Path | None = None
    if template_raw:
        tp = Path(template_raw)
        template_path = repository_relative(root / tp if not tp.is_absolute() else tp, root)

    materialized = dict(scenario_data)
    materialized.pop("pair", None)
    materialized.pop("template", None)
    materialized.update(
        {
            "id": scenario_id,
            "pair_id": pair_id,
            "name": scenario_data.get("name") or scenario_id,
            "market_files": market_files,
            "template_path": template_path.as_posix() if template_path else None,
            "source_path": repository_relative(candidate, root).as_posix(),
        }
    )
    return ScenarioSpec.from_dict(materialized)


__all__ = ["MarketFileRef", "ScenarioSpec", "load_scenario_spec"]
