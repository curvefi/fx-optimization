"""Masked metric resolution shared by heatmap rendering and the CLI."""

from __future__ import annotations

from collections.abc import Collection

MASKED_METRIC_SOURCES = {
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


def masked_metric_source(name: str, available: Collection[str]) -> str | None:
    """Resolve specialized aliases or a literal ``X_masked`` -> ``X`` suffix."""
    specialized = MASKED_METRIC_SOURCES.get(name)
    if specialized is not None:
        return specialized
    if name.endswith("_masked"):
        source = name.removesuffix("_masked")
        if source in available:
            return source
    return None


def is_masked_metric(name: str, available: Collection[str]) -> bool:
    return masked_metric_source(name, available) is not None


def masked_metric_uses_detach(name: str, available: Collection[str]) -> bool:
    """Generic suffix masks use core 7dpdif + detachment semantics."""
    return name not in MASKED_METRIC_SOURCES and is_masked_metric(name, available)

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

# Specialized coupled/YB aliases may also honor skew and final-price thresholds.
SKEW_MASKED_METRICS = frozenset(
    MASKED_METRICS
    - PDIFF_ONLY_MASKED_METRICS
    - set(SLIPPAGE_APY_MASK_SOURCES)
)

__all__ = [
    "MASKED_METRIC_SOURCES",
    "PDIFF_ONLY_MASKED_METRICS",
    "SLIPPAGE_APY_MASK_SOURCES",
    "MASKED_METRICS",
    "SKEW_MASKED_METRICS",
    "is_masked_metric",
    "masked_metric_source",
    "masked_metric_uses_detach",
]
