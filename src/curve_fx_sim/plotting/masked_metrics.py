"""Masked metric derivation shared by heatmap rendering and the CLI.

Semantics are the legacy ``plot_heatmap_nd_opt.py`` masks verbatim: every
masked metric is its source metric with NaN wherever ``max_7d_rel_price_diff``
exceeds the price threshold (bps, fractions stored as 1.0 = 100%); the
``apy_N_masked`` family additionally caps on its own slippage metric
(``tw_real_slippage_{1,5,10}pct > slipthr`` -> NaN).  ``slipthr-max`` is a
legacy UI slider range, not a mask bound, and is ignored here.
"""

from __future__ import annotations

MASKED_METRIC_SOURCES = {
    "apy_masked": "apy_net",
    "apy_masked_imbalance": "apy_net",
    "apy_gm_masked": "apy_net_gm",
    "yb_apy_masked": "yb_apy",
    "yb_apy_gm_masked": "yb_apy_gm",
    "tw_real_slippage_1pct_masked": "tw_real_slippage_1pct",
    "tw_real_slippage_5pct_masked": "tw_real_slippage_5pct",
    "tw_real_slippage_10pct_masked": "tw_real_slippage_10pct",
    "apy_1_masked": "apy_net",
    "apy_5_masked": "apy_net",
    "apy_10_masked": "apy_net",
}

# Slippage metrics whose mask is price-difference only (no self-window).
PDIFF_ONLY_MASKED_METRICS = frozenset(
    {
        "tw_real_slippage_1pct_masked",
        "tw_real_slippage_5pct_masked",
        "tw_real_slippage_10pct_masked",
    }
)

# apy_N_masked: cap on the matching slippage metric.
SLIPPAGE_APY_MASK_SOURCES = {
    "apy_1_masked": "tw_real_slippage_1pct",
    "apy_5_masked": "tw_real_slippage_5pct",
    "apy_10_masked": "tw_real_slippage_10pct",
}

MASKED_METRICS = frozenset(MASKED_METRIC_SOURCES)

# Every masked metric except the price-diff-only family and the slippage-capped
# apy_N_masked family also honors the skew threshold when one is set.
SKEW_MASKED_METRICS = frozenset(
    MASKED_METRICS - PDIFF_ONLY_MASKED_METRICS - set(SLIPPAGE_APY_MASK_SOURCES)
)

__all__ = [
    "MASKED_METRIC_SOURCES",
    "PDIFF_ONLY_MASKED_METRICS",
    "SLIPPAGE_APY_MASK_SOURCES",
    "MASKED_METRICS",
    "SKEW_MASKED_METRICS",
]
