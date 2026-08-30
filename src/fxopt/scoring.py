"""Current research score used by adaptive optimization."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import numpy as np


COMBINED_SCORE = "combined_score"
LP_DETACH_SCORE = "lp_detach_score"
LP_CONSISTENCY_METRIC = "apy_net_consistency_90d"
GM_FLOOR = 1e-4
DETACH_ENERGY_WEIGHT = 2.5
COMBINED_SCORE_FORMULA = (
    "2*ln(sqrt(max(apy_net_gm,1e-4)*max(yb_apy_gm,1e-4)))"
    "-2.5*detach_energy_ungated"
)
LP_DETACH_SCORE_FORMULA = (
    "ln(1+apy_net_consistency_90d)-2.5*detach_energy_ungated"
)


def _combined_formula(lp_gm: Any, yb_gm: Any, detach: Any) -> Any:
    return (
        np.log(np.maximum(lp_gm, GM_FLOOR))
        + np.log(np.maximum(yb_gm, GM_FLOOR))
        - DETACH_ENERGY_WEIGHT * detach
    )


def combined_score(metrics: Mapping[str, float]) -> float:
    """Balance LP/YB geometric APY and charge persistent E1.5 detachment."""
    try:
        lp_gm = float(metrics["apy_net_gm"])
        yb_gm = float(metrics["yb_apy_gm"])
        detach_energy = float(metrics["detach_energy_ungated"])
    except KeyError as exc:
        raise ValueError(f"combined_score requires metric {exc.args[0]!r}") from exc

    if not all(math.isfinite(value) for value in (lp_gm, yb_gm, detach_energy)):
        raise ValueError("combined_score inputs must be finite")
    if detach_energy < 0.0:
        raise ValueError("combined_score requires non-negative detach_energy_ungated")

    return float(_combined_formula(lp_gm, yb_gm, detach_energy))


def combined_scores(metrics: Mapping[str, Any]) -> np.ndarray:
    """Vectorized combined score; invalid rows are NaN."""
    try:
        lp_gm = np.asarray(metrics["apy_net_gm"], dtype=float)
        yb_gm = np.asarray(metrics["yb_apy_gm"], dtype=float)
        detach = np.asarray(metrics["detach_energy_ungated"], dtype=float)
    except KeyError as exc:
        raise ValueError(f"combined_score requires metric {exc.args[0]!r}") from exc
    if lp_gm.ndim != 1 or lp_gm.shape != yb_gm.shape or lp_gm.shape != detach.shape:
        raise ValueError("combined_score metrics must be equal-length vectors")

    valid = np.isfinite(lp_gm) & np.isfinite(yb_gm) & np.isfinite(detach) & (detach >= 0.0)
    scores = np.full(lp_gm.shape, np.nan)
    scores[valid] = _combined_formula(lp_gm[valid], yb_gm[valid], detach[valid])
    return scores


def lp_detach_score(metrics: Mapping[str, float]) -> float:
    """Rank consistent no-YB LP growth after charging E1.5 detachment."""
    try:
        lp_consistency = float(metrics[LP_CONSISTENCY_METRIC])
        detach_energy = float(metrics["detach_energy_ungated"])
    except KeyError as exc:
        raise ValueError(f"lp_detach_score requires metric {exc.args[0]!r}") from exc
    if not math.isfinite(lp_consistency) or not math.isfinite(detach_energy):
        raise ValueError("lp_detach_score inputs must be finite")
    if lp_consistency <= -1.0:
        raise ValueError("lp_detach_score requires apy_net_consistency_90d > -1")
    if detach_energy < 0.0:
        raise ValueError("lp_detach_score requires non-negative detach_energy_ungated")
    return math.log1p(lp_consistency) - DETACH_ENERGY_WEIGHT * detach_energy


def lp_detach_scores(metrics: Mapping[str, Any]) -> np.ndarray:
    """Vectorized no-YB LP-detachment score; invalid rows are NaN."""
    try:
        lp_consistency = np.asarray(metrics[LP_CONSISTENCY_METRIC], dtype=float)
        detach = np.asarray(metrics["detach_energy_ungated"], dtype=float)
    except KeyError as exc:
        raise ValueError(f"lp_detach_score requires metric {exc.args[0]!r}") from exc
    if lp_consistency.ndim != 1 or lp_consistency.shape != detach.shape:
        raise ValueError("lp_detach_score metrics must be equal-length vectors")
    valid = (
        np.isfinite(lp_consistency) & np.isfinite(detach) &
        (lp_consistency > -1.0) & (detach >= 0.0)
    )
    scores = np.full(lp_consistency.shape, np.nan)
    scores[valid] = (
        np.log1p(lp_consistency[valid])
        - DETACH_ENERGY_WEIGHT * detach[valid]
    )
    return scores


__all__ = [
    "COMBINED_SCORE",
    "COMBINED_SCORE_FORMULA",
    "DETACH_ENERGY_WEIGHT",
    "GM_FLOOR",
    "LP_CONSISTENCY_METRIC",
    "LP_DETACH_SCORE",
    "LP_DETACH_SCORE_FORMULA",
    "combined_score",
    "combined_scores",
    "lp_detach_score",
    "lp_detach_scores",
]
