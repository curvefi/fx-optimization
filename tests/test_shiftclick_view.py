"""Rolling 7-day YB APR of the Shift-click view and its axis scaling."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from curve_fx_sim.plotting.shiftclick_view import (  # noqa: E402
    SEC_PER_YEAR,
    _make_yb_path,
    _plot_yb_axis,
    _rolling_window_growth_apr,
)

DAY = 86400.0
WEEK = 7.0 * DAY
NAN = np.nan
LOSS_7D = -SEC_PER_YEAR / WEEK  # a total loss over one week, as a simple annual rate


def daily(days: int) -> np.ndarray:
    return np.arange(days + 1, dtype=float) * DAY


# Simple APR (g_end / g_start - 1) * year / dt, start = the latest sample at least one week back; NaN before a week.
CASES = {
    "constant growth": (daily(20), np.full(21, 1.2), np.r_[np.full(7, NAN), np.zeros(14)]),
    "3% weekly growth is 156.43%": (
        daily(28), np.power(1.03, daily(28) / WEEK), np.r_[np.full(7, NAN), np.full(22, 1.5642857142857143)]),
    "losses kept, nonpositive endpoint is a total loss": (
        daily(9), np.r_[np.ones(7), 0.97, 0.0, -0.25], np.r_[np.full(7, NAN), -0.03 * SEC_PER_YEAR / WEEK, LOSS_7D, LOSS_7D]),
    "irregular timestamps": (
        np.array([0.0, 3.0, 8.0, 10.0, 16.0]) * DAY, np.array([1.0, 1.01, 1.02, 1.04, 1.05]),
        np.array([NAN, NAN, 0.02 * SEC_PER_YEAR / (8.0 * DAY), (1.04 / 1.01 - 1.0) * SEC_PER_YEAR / WEEK,
                  (1.05 / 1.02 - 1.0) * SEC_PER_YEAR / (8.0 * DAY)])),
    "insufficient history": (np.linspace(0.0, 6.9 * DAY, 50), np.linspace(1.0, 1.1, 50), np.full(50, NAN)),
}


@pytest.mark.parametrize("t, growth, expected", CASES.values(), ids=CASES.keys())
def test_rolling_7d_simple_apr(t: np.ndarray, growth: np.ndarray, expected: np.ndarray) -> None:
    np.testing.assert_allclose(_rolling_window_growth_apr(t, growth, WEEK), expected, rtol=1e-12, equal_nan=True)


def test_seven_day_spikes_do_not_set_the_axis_scale() -> None:
    t = np.arange(0.0, 120.0 * DAY + 1.0, 3600.0)
    growth = np.power(1.10, t / (365.0 * DAY))   # ~10%/yr everywhere
    spike = (t >= 100.0 * DAY) & (t < 100.5 * DAY)
    growth[spike] *= 1.05                        # a half-day 5% spike: 7d simple APR ~ +260%
    path = _make_yb_path(t, growth, {}, "test", max_points=0)
    assert np.nanmax(path["rolling_apr_7d"]) > 2.0
    fig, ax = plt.subplots()
    try:
        _plot_yb_axis(ax, path)
        lo, hi = ax.get_ylim()
    finally:
        plt.close(fig)
    anchored = np.concatenate([path["rolling_apy"], path["rolling_apy_30d"]]) * 100.0
    anchored = anchored[np.isfinite(anchored)]
    assert lo <= min(0.0, anchored.min()) - 1.0 + 1e-9
    assert hi >= 1.0 and hi < 100.0
