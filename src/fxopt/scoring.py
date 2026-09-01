"""Research scores derived from completed Cartesian grids."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


LP_ROBUST_METRIC = "apy_net_robust_90d"
GM_FLOOR = 1e-4
DETACH_ENERGY_WEIGHT = 2.5


def _combined_formula(lp_gm: Any, yb_gm: Any, detach: Any) -> Any:
    return (
        np.log(np.maximum(lp_gm, GM_FLOOR))
        + np.log(np.maximum(yb_gm, GM_FLOOR))
        - DETACH_ENERGY_WEIGHT * detach
    )


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


def lp_detach_scores(metrics: Mapping[str, Any]) -> np.ndarray:
    """Vectorized no-YB LP-detachment score; invalid rows are NaN."""
    try:
        lp_robust = np.asarray(metrics[LP_ROBUST_METRIC], dtype=float)
        detach = np.asarray(metrics["detach_energy_ungated"], dtype=float)
    except KeyError as exc:
        raise ValueError(f"lp_detach_score requires metric {exc.args[0]!r}") from exc
    if lp_robust.ndim != 1 or lp_robust.shape != detach.shape:
        raise ValueError("lp_detach_score metrics must be equal-length vectors")
    valid = (
        np.isfinite(lp_robust) & np.isfinite(detach) &
        (lp_robust > -1.0) & (detach >= 0.0)
    )
    scores = np.full(lp_robust.shape, np.nan)
    scores[valid] = (
        np.log1p(lp_robust[valid])
        - DETACH_ENERGY_WEIGHT * detach[valid]
    )
    return scores


__all__ = [
    "DETACH_ENERGY_WEIGHT",
    "GM_FLOOR",
    "LP_ROBUST_METRIC",
    "combined_scores",
    "lp_detach_scores",
]
