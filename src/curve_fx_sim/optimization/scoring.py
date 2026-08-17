"""One strict scoring contract for compiled-policy weight optimization."""

from __future__ import annotations

import math
from typing import Any, Sequence

SCORE_FX_LP_E15_SLIPPAGE_V1_KEY = "score_fx_lp_e15_slippage_v1"
SCORE_VERSION = SCORE_FX_LP_E15_SLIPPAGE_V1_KEY

FAIL_SCORE = -10.0
FAIL_LOSS = 25.0

E15_WEIGHT = 2.5
E15_REFERENCE_DAYS = 21.0
SLIPPAGE_WEIGHT = 10.0
SLIPPAGE_TARGET = 0.001
SLIPPAGE_EXCESS_WEIGHT = 100.0


def normalize_score_key(raw: str) -> str:
    """Accept only the product's single explicit objective identifier."""
    key = raw.strip()
    if key != SCORE_FX_LP_E15_SLIPPAGE_V1_KEY:
        raise ValueError(
            f"unsupported score key {raw!r}; expected {SCORE_FX_LP_E15_SLIPPAGE_V1_KEY!r}"
        )
    return key


def _finite(value: Any, *, minimum: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        return None
    return result


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def score_scenarios(
    scenarios: list[dict[str, Any]],
    *,
    require_yb: bool = False,
) -> dict[str, Any]:
    """Score all scenarios with the attested LP/E15/slippage objective.

    YieldBasis metrics are an eligibility gate only.  They never silently
    select a second economic objective.
    """
    values: list[float] = []
    e15_penalties: list[float] = []
    slippage_penalties: list[float] = []
    detach_values: list[float] = []
    slippage_values: list[float] = []
    failures = 0
    yb_failures = 0

    for scenario in scenarios:
        apy = _finite(scenario.get("apy_net"), minimum=-0.999999)
        detach = _finite(
            scenario.get("detach_energy_ungated", scenario.get("detach_energy")),
            minimum=0.0,
        )
        duration = _finite(scenario.get("duration_s"), minimum=0.0)
        slippage = _finite(
            scenario.get("tw_real_slippage_1pct", scenario.get("real_slippage_1pct")),
            minimum=0.0,
        )
        yb_ok = True
        if require_yb:
            yb_ok = (
                not bool(scenario.get("yb_wiped"))
                and _finite(scenario.get("yb_apy"), minimum=-0.999999) is not None
                and _finite(scenario.get("yb_apy_gm"), minimum=-0.999999) is not None
            )
            if not yb_ok:
                yb_failures += 1

        if (
            not bool(scenario.get("ok", True))
            or apy is None
            or detach is None
            or duration is None
            or duration <= 0.0
            or slippage is None
            or not yb_ok
        ):
            failures += 1
            continue

        replay_days = max(1.0, duration / 86400.0)
        e15_penalty = E15_WEIGHT * E15_REFERENCE_DAYS * detach / replay_days
        slippage_penalty = (
            SLIPPAGE_WEIGHT * slippage
            + SLIPPAGE_EXCESS_WEIGHT * max(0.0, slippage - SLIPPAGE_TARGET)
        )
        values.append(apy - e15_penalty - slippage_penalty)
        e15_penalties.append(e15_penalty)
        slippage_penalties.append(slippage_penalty)
        detach_values.append(detach)
        slippage_values.append(slippage)

    objective = FAIL_SCORE if failures or not values else float(_mean(values))
    return {
        "score_version": SCORE_VERSION,
        SCORE_FX_LP_E15_SLIPPAGE_V1_KEY: objective,
        "gate": failures == 0 and bool(values),
        "mean_e15_penalty": _mean(e15_penalties),
        "mean_slippage_penalty": _mean(slippage_penalties),
        "mean_detach_energy_ungated": _mean(detach_values),
        "mean_tw_real_slippage_1pct": _mean(slippage_values),
        "n_scenarios": len(scenarios),
        "n_failed": failures,
        "n_yb_failed": yb_failures,
    }


def score_objective_value(score: dict[str, Any], score_key: str) -> float:
    """Extract the one supported finite objective from a score payload."""
    key = normalize_score_key(score_key)
    if key not in score:
        raise KeyError(f"score key {key!r} missing from score summary")
    value = float(score[key])
    if not math.isfinite(value):
        raise ValueError(f"score key {key!r} has non-finite value: {score[key]!r}")
    return value


def objective_failure_count(score: dict[str, Any], score_key: str) -> int:
    """Count scenarios ineligible for the selected objective."""
    normalize_score_key(score_key)
    return int(score.get("n_failed", 0))


def loss_from_score(score: dict[str, Any]) -> float:
    """Return minimization loss for a score whose objective was selected."""
    if int(score.get("objective_failures", score.get("n_failed", 0))) > 0:
        return FAIL_LOSS
    value = _finite(score.get("objective_value"))
    return -value if value is not None else FAIL_LOSS


__all__ = [
    "FAIL_LOSS",
    "FAIL_SCORE",
    "SCORE_FX_LP_E15_SLIPPAGE_V1_KEY",
    "SCORE_VERSION",
    "loss_from_score",
    "normalize_score_key",
    "objective_failure_count",
    "score_objective_value",
    "score_scenarios",
]
