"""Evaluation statistics."""

from __future__ import annotations

import numpy as np
import pytest

from hddlab.metrics import MDE_MULTIPLIER, coverage, mean_estimate, newey_west_se, standardised_rms


def test_newey_west_without_lags_is_naive_se() -> None:
    x = np.random.default_rng(0).normal(size=50)
    assert newey_west_se(x, 0) == pytest.approx(x.std(ddof=0) / np.sqrt(50))


def test_newey_west_larger_for_positively_correlated_series() -> None:
    rng = np.random.default_rng(1)
    x = np.zeros(400)
    for t in range(1, 400):
        x[t] = 0.6 * x[t - 1] + rng.normal()
    # True long-run ratio is sqrt((1 + 0.6) / (1 - 0.6)) = 2; Bartlett weights with 5 lags undershoot it.
    assert newey_west_se(x, 5) > 1.3 * newey_west_se(x, 0)


def test_mde_multiplier() -> None:
    assert MDE_MULTIPLIER == pytest.approx(2.8016, abs=1e-4)
    est = mean_estimate(np.array([1.0, -1.0, 1.0, -1.0]), 0)
    assert est.mde == pytest.approx(MDE_MULTIPLIER * est.se)


def test_coverage_z() -> None:
    cov = coverage(np.array([True] * 45 + [False] * 5), 0.9)
    assert cov.rate == pytest.approx(0.9)
    assert cov.z == pytest.approx(0.0)


def test_standardised_rms_of_calibrated_forecasts_is_one() -> None:
    z = np.random.default_rng(2).normal(size=100_000)
    value, se = standardised_rms(z)
    assert value == pytest.approx(1.0, abs=4 * se)
