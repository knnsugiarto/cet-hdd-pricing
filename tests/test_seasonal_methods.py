"""Burn analysis, index modelling and reference strikes."""

from __future__ import annotations

import numpy as np
import pytest

from hddlab.seasonal_methods import burn_forecast, index_forecast, reference_strikes, trend_adjusted


def test_linear_detrend_projects_to_target_year() -> None:
    years = np.arange(1990, 2020)
    totals = 2000.0 - 3.0 * (years - 1990)  # exactly linear: no residuals
    adjusted = trend_adjusted(totals, years, 2020, "linear")
    assert np.allclose(adjusted, 2000.0 - 3.0 * 30)


def test_no_trend_leaves_totals() -> None:
    totals = np.array([1.0, 5.0, 3.0])
    assert np.array_equal(trend_adjusted(totals, np.arange(3), 10, "none"), totals)


def test_burn_is_empirical_distribution() -> None:
    totals = np.array([1800.0, 1900.0, 2000.0, 2100.0])
    fc = burn_forecast(totals, np.arange(4), 4, "none")
    assert fc.mean == pytest.approx(1950.0)
    # Burn price of a call struck at 1950: mean of payoffs (0, 0, 50, 150).
    assert fc.option_moments("call", 1950.0)[0] == pytest.approx(50.0)


def test_index_sd_includes_estimation_error() -> None:
    rng = np.random.default_rng(0)
    totals = rng.normal(1900, 150, size=30)
    years = np.arange(1975, 2005)
    flat = index_forecast(totals, years, 2005, "none")
    assert flat.sd == pytest.approx(totals.std(ddof=1) * np.sqrt(1 + 1 / 30))
    trend = index_forecast(totals, years, 2005, "linear")
    slope, intercept = np.polyfit(years, totals, 1)
    resid = totals - (intercept + slope * years)
    s = np.sqrt(np.sum(resid**2) / 28)
    sxx = np.sum((years - years.mean()) ** 2)
    assert trend.mean == pytest.approx(intercept + slope * 2005)
    assert trend.sd == pytest.approx(s * np.sqrt(1 + 1 / 30 + (2005 - years.mean()) ** 2 / sxx))


def test_reference_strikes() -> None:
    totals = np.array([1800.0, 2000.0, 2200.0])
    strikes = reference_strikes(totals, (-1.0, 0.0, 1.0))
    assert strikes[0.0] == pytest.approx(2000.0)
    assert strikes[1.0] - strikes[0.0] == pytest.approx(200.0)
    assert strikes[-1.0] == pytest.approx(1800.0)
