"""Generic masked metric resolution for heatmap views."""

from __future__ import annotations

from collections.abc import Collection, Sequence


_SOURCE_ALIASES = {
    "apy_masked": "apy_net",
    "apy_1_masked": "apy_net",
    "apy_5_masked": "apy_net",
}

_SLIPPAGE_APY_SOURCES = {
    "apy_1_masked": "tw_real_slippage_1pct",
    "apy_5_masked": "tw_real_slippage_5pct",
}


def masked_metric_source(name: str, available: Collection[str]) -> str | None:
    """Resolve ``X_masked`` to ``X`` when the raw metric is available."""
    source = _SOURCE_ALIASES.get(name)
    if source is None and name.endswith("_masked"):
        source = name.removesuffix("_masked")
    if source is None:
        return None
    return source if source in available else None


def is_masked_metric(name: str, available: Collection[str]) -> bool:
    return masked_metric_source(name, available) is not None


def masked_metric_uses_detach(name: str, available: Collection[str]) -> bool:
    return (
        is_masked_metric(name, available)
        and name not in _SLIPPAGE_APY_SOURCES
        and not name.startswith("tw_real_slippage_")
    )


def masked_metric_slippage_source(name: str) -> str | None:
    return _SLIPPAGE_APY_SOURCES.get(name)


def masked_metric_slippage_sources(
    names: Sequence[str], available: Collection[str]
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            source
            for name in names
            if (source := masked_metric_slippage_source(name)) in available
        )
    )


def masked_metric_uses_slippage(name: str, available: Collection[str]) -> bool:
    return (
        is_masked_metric(name, available)
        and name != "apy_masked"
        and not name.startswith("tw_real_slippage_")
    )


__all__ = [
    "is_masked_metric",
    "masked_metric_slippage_source",
    "masked_metric_slippage_sources",
    "masked_metric_source",
    "masked_metric_uses_detach",
    "masked_metric_uses_slippage",
]
