"""Deterministic tests for the single product scoring contract."""

import math

import pytest

from curve_fx_sim.optimization.scoring import (
    FAIL_LOSS,
    FAIL_SCORE,
    SCORE_FX_LP_E15_SLIPPAGE_V1_KEY,
    loss_from_score,
    normalize_score_key,
    score_objective_value,
    score_scenarios,
)


def _scenario() -> dict[str, float | bool]:
    return {
        "ok": True,
        "apy_net": 0.05,
        "tw_real_slippage_1pct": 0.0005,
        "detach_energy_ungated": 0.001,
        "duration_s": 21 * 86400.0,
    }


def test_fx_lp_e15_slippage_formula() -> None:
    result = score_scenarios([_scenario()])
    assert result["score_version"] == SCORE_FX_LP_E15_SLIPPAGE_V1_KEY
    assert result["gate"] is True
    assert result["n_failed"] == 0
    # E15 penalty: 2.5 * 21 * .001 / 21 = .0025.
    # Slippage penalty: 10 * .0005 = .005.
    assert math.isclose(
        result[SCORE_FX_LP_E15_SLIPPAGE_V1_KEY],
        0.0425,
        rel_tol=1e-12,
    )


def test_score_key_is_exact_and_loss_uses_selected_objective() -> None:
    score = {
        SCORE_FX_LP_E15_SLIPPAGE_V1_KEY: 0.0425,
        "objective_value": 0.0425,
        "n_failed": 0,
    }
    assert normalize_score_key(SCORE_FX_LP_E15_SLIPPAGE_V1_KEY) == SCORE_FX_LP_E15_SLIPPAGE_V1_KEY
    assert score_objective_value(score, SCORE_FX_LP_E15_SLIPPAGE_V1_KEY) == 0.0425
    assert loss_from_score(score) == -0.0425
    with pytest.raises(ValueError, match="unsupported score key"):
        normalize_score_key("fx_score")


def test_yb_requirement_is_an_eligibility_gate() -> None:
    result = score_scenarios([_scenario()], require_yb=True)
    assert result["gate"] is False
    assert result["n_failed"] == 1
    assert result["n_yb_failed"] == 1
    assert result[SCORE_FX_LP_E15_SLIPPAGE_V1_KEY] == FAIL_SCORE
    result["objective_value"] = FAIL_SCORE
    assert loss_from_score(result) == FAIL_LOSS


def test_failed_evaluation_uses_bounded_sentinel() -> None:
    result = score_scenarios([{"ok": False, "error": "revert"}])
    assert result["gate"] is False
    assert result["n_failed"] == 1
    assert result[SCORE_FX_LP_E15_SLIPPAGE_V1_KEY] == FAIL_SCORE
