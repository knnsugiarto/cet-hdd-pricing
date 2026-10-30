"""Gaussian closed forms against numerical integration and their own identities."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate
from scipy.stats import norm

from hddlab.distributions import (
    Forecast,
    bachelier_call,
    bachelier_put,
    call_second_moment,
    ensemble_crps,
    expected_hdd_gaussian,
    gaussian_crps,
    put_second_moment,
)

CASES = [(1900.0, 150.0, 1900.0), (1900.0, 150.0, 2050.0), (1900.0, 150.0, 1700.0), (5.0, 0.3, 4.0)]


def _integral(f, mean: float, sd: float) -> float:
    value, _ = integrate.quad(lambda h: f(h) * norm.pdf(h, mean, sd), mean - 12 * sd, mean + 12 * sd, limit=200)
    return value


@pytest.mark.parametrize(("m", "s", "k"), CASES)
def test_bachelier_matches_integration(m: float, s: float, k: float) -> None:
    assert bachelier_call(m, s, k) == pytest.approx(_integral(lambda h: max(h - k, 0.0), m, s), rel=1e-7, abs=1e-9)
    assert bachelier_put(m, s, k) == pytest.approx(_integral(lambda h: max(k - h, 0.0), m, s), rel=1e-7, abs=1e-9)


@pytest.mark.parametrize(("m", "s", "k"), CASES)
def test_put_call_parity(m: float, s: float, k: float) -> None:
    assert bachelier_call(m, s, k) - bachelier_put(m, s, k) == pytest.approx(m - k)


@pytest.mark.parametrize(("m", "s", "k"), CASES)
def test_second_moments(m: float, s: float, k: float) -> None:
    # abs tolerance covers quadrature error on deep out-of-the-money tails (values ~1e-6)
    tol = {"rel": 1e-7, "abs": 1e-10}
    assert call_second_moment(m, s, k) == pytest.approx(_integral(lambda h: max(h - k, 0.0) ** 2, m, s), **tol)
    assert put_second_moment(m, s, k) == pytest.approx(_integral(lambda h: max(k - h, 0.0) ** 2, m, s), **tol)


def test_expected_hdd_gaussian() -> None:
    mean, var = np.array([5.0, 17.0, 25.0]), np.array([4.0, 9.0, 1.0])
    expected = [_integral(lambda t: max(18.0 - t, 0.0), m, np.sqrt(v)) for m, v in zip(mean, var)]
    assert np.allclose(expected_hdd_gaussian(18.0, mean, var), expected, rtol=1e-7, atol=1e-12)


def test_gaussian_crps_matches_integral_definition() -> None:
    # CRPS = integral of (F(x) - 1{x >= y})^2 dx.
    m, s, y = 1900.0, 150.0, 2010.0
    lo, hi = m - 12 * s, m + 12 * s
    left, _ = integrate.quad(lambda x: norm.cdf(x, m, s) ** 2, lo, y, limit=200)
    right, _ = integrate.quad(lambda x: (1 - norm.cdf(x, m, s)) ** 2, y, hi, limit=200)
    assert gaussian_crps(m, s, y) == pytest.approx(left + right, rel=1e-7)


def test_ensemble_crps_matches_brute_force() -> None:
    rng = np.random.default_rng(1)
    x = np.sort(rng.normal(size=37))
    y = 0.3
    brute = np.mean(np.abs(x - y)) - 0.5 * np.mean(np.abs(x[:, None] - x[None, :]))
    assert ensemble_crps(x, y) == pytest.approx(brute, rel=1e-12)


def test_large_ensemble_approaches_gaussian() -> None:
    rng = np.random.default_rng(2)
    samples = rng.normal(1900.0, 150.0, size=200_000)
    ens = Forecast.ensemble(samples)
    gau = Forecast.gaussian(1900.0, 150.0)
    assert ens.crps(2000.0) == pytest.approx(gau.crps(2000.0), rel=0.01)
    assert ens.option_moments("call", 1950.0)[0] == pytest.approx(gau.option_moments("call", 1950.0)[0], rel=0.01)
    assert ens.quantile(0.9) == pytest.approx(gau.quantile(0.9), rel=0.002)


def test_ensemble_pit_counts_ties_half() -> None:
    fc = Forecast.ensemble(np.array([1.0, 2.0, 2.0, 3.0]))
    assert fc.cdf(2.0) == pytest.approx(0.5)
    assert fc.cdf(0.0) == 0.0
    assert fc.cdf(5.0) == 1.0
