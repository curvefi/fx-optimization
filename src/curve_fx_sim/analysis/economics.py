"""Strict deterministic comparison of summary and replay economics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


class EconomicMismatch(ValueError):
    """Raised when a full-observation replay differs from summary economics."""


@dataclass(frozen=True)
class MetricComparison:
    field: str
    expected: float
    observed: float
    absolute_error: float
    relative_error: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "expected": self.expected,
            "observed": self.observed,
            "absolute_error": self.absolute_error,
            "relative_error": self.relative_error,
        }


@dataclass(frozen=True)
class EconomicComparison:
    fingerprint: str
    metrics: tuple[MetricComparison, ...]
    relative_tolerance: float
    absolute_tolerance: float

    @property
    def max_absolute_error(self) -> float:
        return max((item.absolute_error for item in self.metrics), default=0.0)

    @property
    def max_relative_error(self) -> float:
        return max((item.relative_error for item in self.metrics), default=0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "metric_count": len(self.metrics),
            "relative_tolerance": self.relative_tolerance,
            "absolute_tolerance": self.absolute_tolerance,
            "max_absolute_error": self.max_absolute_error,
            "max_relative_error": self.max_relative_error,
            "metrics": [item.to_dict() for item in self.metrics],
        }


_DEFAULT_EXCLUDED_FIELDS = frozenset({"elapsed_ms"})


def compare_economics(
    expected_metrics: Mapping[str, Any],
    observed_metrics: Mapping[str, Any],
    *,
    expected_fingerprint: str,
    observed_fingerprint: str,
    fields: Sequence[str] | None = None,
    relative_tolerance: float = 1e-12,
    absolute_tolerance: float = 0.0,
) -> EconomicComparison:
    """Validate economic fingerprint and every projected deterministic metric."""
    if not expected_fingerprint:
        raise EconomicMismatch("source selection has no economic fingerprint")
    if not observed_fingerprint:
        raise EconomicMismatch("replay result has no economic fingerprint")
    if expected_fingerprint != observed_fingerprint:
        raise EconomicMismatch(
            "economic fingerprint mismatch: "
            f"summary={expected_fingerprint!r}, replay={observed_fingerprint!r}"
        )
    if relative_tolerance < 0 or absolute_tolerance < 0:
        raise ValueError("economic comparison tolerances must be nonnegative")

    selected_fields = tuple(fields) if fields is not None else tuple(sorted(expected_metrics))
    selected_fields = tuple(
        field for field in selected_fields if field not in _DEFAULT_EXCLUDED_FIELDS
    )
    if not selected_fields:
        raise EconomicMismatch("source selection has no deterministic projected metrics")

    comparisons: list[MetricComparison] = []
    mismatches: list[str] = []
    for field in selected_fields:
        if field not in expected_metrics:
            raise EconomicMismatch(f"summary metric projection is missing {field!r}")
        if field not in observed_metrics:
            raise EconomicMismatch(f"replay result is missing projected metric {field!r}")
        try:
            expected = float(expected_metrics[field])
            observed = float(observed_metrics[field])
        except (TypeError, ValueError) as exc:
            raise EconomicMismatch(f"metric {field!r} is not numeric") from exc
        if not math.isfinite(expected) or not math.isfinite(observed):
            raise EconomicMismatch(f"metric {field!r} is not finite")
        absolute_error = abs(observed - expected)
        denominator = max(abs(expected), abs(observed))
        relative_error = absolute_error / denominator if denominator else 0.0
        comparison = MetricComparison(
            field=field,
            expected=expected,
            observed=observed,
            absolute_error=absolute_error,
            relative_error=relative_error,
        )
        comparisons.append(comparison)
        if not math.isclose(
            observed,
            expected,
            rel_tol=relative_tolerance,
            abs_tol=absolute_tolerance,
        ):
            mismatches.append(
                f"{field}: summary={expected!r}, replay={observed!r}, "
                f"abs={absolute_error:.17g}, rel={relative_error:.17g}"
            )
    if mismatches:
        raise EconomicMismatch("economic metric mismatch: " + "; ".join(mismatches))
    return EconomicComparison(
        fingerprint=observed_fingerprint,
        metrics=tuple(comparisons),
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )


__all__ = [
    "EconomicComparison",
    "EconomicMismatch",
    "MetricComparison",
    "compare_economics",
]
