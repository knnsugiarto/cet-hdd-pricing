"""The daily model: fitting, the simulator's moments, and the exact covariance propagation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.ar_model import AutoReg

from hddlab.daily_model import (
    DailyModel,
    DailySpec,
    fit_ar,
    fit_daily_model,
    fourier_design,
    gaussian_moments,
    simulate_index,
    years_since,
)


def _ar1_model(phi: float, sigma: float, x0: float, origin: pd.Timestamp, level: float = 5.0) -> DailyModel:
    """A hand-built AR(1) model with a flat seasonal mean, for analytic checks."""
    spec = DailySpec(window_years=1, harmonics=1, ar_order=1, vol="constant")
    return DailyModel(
        spec=spec,
        origin=origin,
        mean_coef=np.array([level, 0.0, 0.0, 0.0]),
        phi=np.array([phi]),
        vol_coef=np.array([sigma**2]),
        vol_floor=0.0,
        last_anomalies=np.array([x0]),
        resid_acf=np.zeros(5),
        resid_sd=sigma,
        residuals=np.zeros(1),
    )


def test_ar_fit_matches_statsmodels() -> None:
    rng = np.random.default_rng(3)
    x = np.zeros(5000)
    for t in range(3, x.size):
        x[t] = 0.7 * x[t - 1] - 0.1 * x[t - 2] + 0.05 * x[t - 3] + rng.normal()
    for order in (1, 3):
        phi, _ = fit_ar(x, order)
        ref = AutoReg(x, lags=order, trend="n").fit()
        assert np.allclose(phi, ref.params, atol=1e-10)


def test_seasonal_mean_recovered_without_noise() -> None:
    dates = pd.date_range("1990-01-01", "2009-12-31", freq="D")
    origin = pd.Timestamp("2010-01-01")
    coef = np.array([10.0, 0.02, -6.5, 1.2, 0.3, -0.2])
    temps = pd.Series(fourier_design(years_since(dates, origin), 2, True) @ coef, index=dates)
    model = fit_daily_model(temps, DailySpec(20, 2, 1, "constant"))
    assert np.allclose(model.mean_coef, coef, atol=1e-8)


def test_window_longer_than_data_raises() -> None:
    temps = pd.Series(5.0, index=pd.date_range("2000-01-01", "2009-12-31", freq="D"))
    with pytest.raises(ValueError):
        fit_daily_model(temps, DailySpec(30, 2, 1, "constant"))


def test_simulator_matches_analytic_ar1_moments() -> None:
    """Mean phi^t x0 and variance sigma^2 (1 - phi^(2t)) / (1 - phi^2) of the OU discretisation."""
    phi, sigma, x0, base = 0.8, 2.0, 3.0, 18.0
    origin = pd.Timestamp("2020-11-01")
    model = _ar1_model(phi, sigma, x0, origin)
    n_days, n_paths = 30, 200_000
    dates = pd.date_range(origin, periods=n_days, freq="D")
    eps = np.random.default_rng(4).standard_normal((n_paths, n_days))
    for day in (1, 5, 30):
        mask = np.zeros(n_days, dtype=bool)
        mask[day - 1] = True
        lin = simulate_index(model, dates[:n_days], mask, (base,), eps)[base].l_samples
        x = (base - 5.0) - lin  # the simulated anomaly on that day
        mean_exact = phi**day * x0
        var_exact = sigma**2 * (1 - phi ** (2 * day)) / (1 - phi**2)
        assert abs(x.mean() - mean_exact) < 4 * np.sqrt(var_exact / n_paths)
        assert x.var() == pytest.approx(var_exact, rel=4 * np.sqrt(2 / n_paths))


def test_propagated_variance_matches_explicit_covariance_sum() -> None:
    """Var(sum X_t) from the state recursion equals the sum of the analytic covariance matrix."""
    phi, sigma, x0 = 0.75, 1.7, -2.0
    origin = pd.Timestamp("2020-11-01")
    model = _ar1_model(phi, sigma, x0, origin)
    n = 40
    dates = pd.date_range(origin, periods=n, freq="D")
    mask = np.zeros(n, dtype=bool)
    mask[5:] = True  # first five days simulated but not summed
    moments = gaussian_moments(model, dates, mask)
    i = np.arange(1, n + 1)
    lo = np.minimum.outer(i, i)
    cov = sigma**2 * phi ** np.abs(np.subtract.outer(i, i)) * (1 - phi ** (2 * lo)) / (1 - phi**2)
    assert moments.sum_anomaly_var == pytest.approx(cov[np.ix_(mask, mask)].sum(), rel=1e-12)
    assert moments.sum_anomaly_mean == pytest.approx(np.sum(phi ** i[mask] * x0), rel=1e-12)
    assert np.allclose(moments.day_var_temp, np.diag(cov), rtol=1e-12)


def test_ar3_seasonal_vol_moments_match_simulation() -> None:
    """For AR(3) with seasonal volatility, the exact moments agree with Monte Carlo."""
    rng = np.random.default_rng(5)
    dates = pd.date_range("1980-11-01", "2020-10-31", freq="D")
    t = years_since(dates, pd.Timestamp("2020-11-01"))
    sig = 2.0 + 0.5 * np.cos(2 * np.pi * t)
    x = np.zeros(len(dates))
    for k in range(3, x.size):
        x[k] = 0.85 * x[k - 1] - 0.15 * x[k - 2] + 0.05 * x[k - 3] + sig[k] * rng.normal()
    temps = pd.Series(10.0 - 6.0 * np.cos(2 * np.pi * t) + x, index=dates)
    model = fit_daily_model(temps, DailySpec(30, 2, 3, "seasonal"))
    future = pd.date_range(model.origin, "2021-03-31", freq="D")
    mask = np.ones(len(future), dtype=bool)
    eps = np.random.default_rng(6).standard_normal((100_000, len(future)))
    sims = simulate_index(model, future, mask, (18.0,), eps)[18.0]
    moments = gaussian_moments(model, future, mask)
    mean_l, sd_l = moments.linear_index(18.0, mask)
    n = sims.l_samples.size
    assert abs(sims.l_samples.mean() - mean_l) < 4 * sd_l / np.sqrt(n)
    assert sims.l_samples.std() == pytest.approx(sd_l, rel=4 * np.sqrt(2 / n))
