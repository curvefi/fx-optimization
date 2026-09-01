"""Generic masked metric resolution for heatmap views."""

from __future__ import annotations

from collections.abc import Collection


def masked_metric_source(name: str, available: Collection[str]) -> str | None:
    """Resolve ``X_masked`` to ``X`` when the raw metric is available."""
    if not name.endswith("_masked"):
        return None
    source = name.removesuffix("_masked")
    return source if source in available else None


def is_masked_metric(name: str, available: Collection[str]) -> bool:
    return masked_metric_source(name, available) is not None


def masked_metric_uses_detach(name: str, available: Collection[str]) -> bool:
    return is_masked_metric(name, available)


__all__ = ["is_masked_metric", "masked_metric_source", "masked_metric_uses_detach"]
