"""Current research score used by adaptive optimization."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import numpy as np


COMBINED_SCORE = "combined_score"
GM_FLOOR = 1e-4
DETACH_ENERGY_WEIGHT = 2.5
COMBINED_SCORE_FORMULA = (
    "2*ln(sqrt(max(apy_net_gm,1e-4)*max(yb_apy_gm,1e-4)))"
    "-2.5*detach_energy_ungated"
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


__all__ = [
    "COMBINED_SCORE",
    "COMBINED_SCORE_FORMULA",
    "DETACH_ENERGY_WEIGHT",
    "GM_FLOOR",
    "combined_score",
    "combined_scores",
]
