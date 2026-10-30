"""Daily temperature model: seasonal mean plus an autoregressive anomaly.

    T_t = s(t) + X_t
    s(t) = a + b t + sum_{k=1..K} [c_k cos(2 pi k t) + d_k sin(2 pi k t)]   (t in years)
    X_t = phi_1 X_{t-1} + ... + phi_p X_{t-p} + sigma(t) eps_t,  eps_t iid N(0, 1)

With p = 1 this is the exact discretisation of an Ornstein-Uhlenbeck process sampled
daily: phi = exp(-kappa dt). sigma(t) is either constant or a Fourier series in the
day of year fitted to squared AR residuals (Benth and Saltyte-Benth's approach).

Everything is fitted by least squares on the ``window_years`` of daily data before
the pricing date, and the simulation starts from the last ``p`` observed anomalies.
Because X is a linear Gaussian process, the sum L = sum(base - T_t) over the season
is exactly Gaussian, and its mean and variance follow from propagating the state's
mean and covariance (``gaussian_moments``). H = sum max(base - T_t, 0) >= L is not
Gaussian, so the Monte Carlo on H is checked against the closed form on L, and the
exact E[H] (a sum of per-day Gaussian expectations).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from hddlab.distributions import expected_hdd_gaussian

DAYS_PER_YEAR = 365.2425


@dataclass(frozen=True)
class DailySpec:
    window_years: int
    harmonics: int
    ar_order: int
    vol: str  # "constant" or "seasonal"
    vol_harmonics: int = 2

    def __post_init__(self) -> None:
        if self.vol not in ("constant", "seasonal"):
            raise ValueError(f"Unknown volatility form {self.vol!r}.")
        if self.ar_order < 1:
            raise ValueError("AR order must be at least 1.")

    @property
    def label(self) -> str:
        return (
            f"daily|ar={self.ar_order}|harmonics={self.harmonics}"
            f"|vol={self.vol}|window={self.window_years}"
        )


def years_since(dates: pd.DatetimeIndex, origin: pd.Timestamp) -> np.ndarray:
    """Time in years (of 365.2425 days) from ``origin``; negative before it."""
    return (dates - origin).days.to_numpy(dtype=float) / DAYS_PER_YEAR


def fourier_design(t: np.ndarray, harmonics: int, with_trend: bool) -> np.ndarray:
    """Columns [1, (t), cos(2 pi k t), sin(2 pi k t) for k = 1..harmonics]."""
    cols = [np.ones_like(t)]
    if with_trend:
        cols.append(t)
    for k in range(1, harmonics + 1):
        cols.append(np.cos(2.0 * np.pi * k * t))
        cols.append(np.sin(2.0 * np.pi * k * t))
    return np.column_stack(cols)


def fit_ar(x: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares AR(order) fit without intercept. Returns (phi, residuals)."""
    n = x.size
    target = x[order:]
    lags = np.column_stack([x[order - i : n - i] for i in range(1, order + 1)])
    phi, *_ = np.linalg.lstsq(lags, target, rcond=None)
    return phi, target - lags @ phi


def autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Sample autocorrelation at lags 1..max_lag."""
    x = x - x.mean()
    denom = np.dot(x, x)
    return np.array([np.dot(x[:-k], x[k:]) / denom for k in range(1, max_lag + 1)])


def companion(phi: np.ndarray) -> np.ndarray:
    """Companion matrix of an AR(p): state (X_t, X_{t-1}, ..., X_{t-p+1})."""
    p = phi.size
    a = np.zeros((p, p))
    a[0, :] = phi
    if p > 1:
        a[1:, :-1] = np.eye(p - 1)
    return a


@dataclass
class DailyModel:
    """A fitted daily model, ready to simulate from ``origin`` (the first unobserved day)."""

    spec: DailySpec
    origin: pd.Timestamp
    mean_coef: np.ndarray
    phi: np.ndarray
    vol_coef: np.ndarray
    vol_floor: float
    last_anomalies: np.ndarray  # most recent first: X_{origin-1}, X_{origin-2}, ...
    resid_acf: np.ndarray
    resid_sd: float
    residuals: np.ndarray

    def seasonal_mean(self, dates: pd.DatetimeIndex) -> np.ndarray:
        t = years_since(dates, self.origin)
        return fourier_design(t, self.spec.harmonics, with_trend=True) @ self.mean_coef

    def sigma(self, dates: pd.DatetimeIndex) -> np.ndarray:
        if self.spec.vol == "constant":
            return np.full(len(dates), np.sqrt(self.vol_coef[0]))
        t = years_since(dates, self.origin)
        var = fourier_design(t, self.spec.vol_harmonics, with_trend=False) @ self.vol_coef
        return np.sqrt(np.maximum(var, self.vol_floor))


def fit_daily_model(temps_before: pd.Series, spec: DailySpec, acf_lags: int = 5) -> DailyModel:
    """Fit the model on the ``spec.window_years`` years before the day after the last observation.

    ``temps_before`` must end on the last day available at the pricing date; nothing
    after it is used. Raises ValueError if the series is shorter than the window.
    """
    origin = temps_before.index[-1] + pd.Timedelta(days=1)
    window_start = origin - pd.DateOffset(years=spec.window_years)
    if temps_before.index[0] > window_start:
        raise ValueError(
            f"Need data from {window_start.date()}; series starts {temps_before.index[0].date()}."
        )
    window = temps_before.loc[window_start:]
    t = years_since(window.index, origin)
    y = window.to_numpy(dtype=float)

    design = fourier_design(t, spec.harmonics, with_trend=True)
    mean_coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    anomalies = y - design @ mean_coef

    phi, resid = fit_ar(anomalies, spec.ar_order)
    resid_sq = resid**2
    if spec.vol == "constant":
        vol_coef = np.array([resid_sq.mean()])
    else:
        vol_design = fourier_design(t[spec.ar_order :], spec.vol_harmonics, with_trend=False)
        vol_coef, *_ = np.linalg.lstsq(vol_design, resid_sq, rcond=None)
    return DailyModel(
        spec=spec,
        origin=origin,
        mean_coef=mean_coef,
        phi=phi,
        vol_coef=vol_coef,
        vol_floor=0.1 * float(resid_sq.mean()),
        last_anomalies=anomalies[-spec.ar_order :][::-1].copy(),
        resid_acf=autocorrelation(resid, acf_lags),
        resid_sd=float(resid.std()),
        residuals=resid,
    )


@dataclass
class SimulatedIndex:
    """Monte Carlo draws of H and of the linear index L for one base temperature."""

    h_samples: np.ndarray
    l_samples: np.ndarray


def simulate_index(
    model: DailyModel,
    future_dates: pd.DatetimeIndex,
    season_mask: np.ndarray,
    bases: tuple[float, ...],
    eps: np.ndarray,
) -> dict[float, SimulatedIndex]:
    """Simulate daily temperatures from ``model.origin`` and accumulate H and L per base.

    ``future_dates`` must start at ``model.origin`` and run to the season end;
    ``season_mask`` marks the days that count towards the index (days before the
    season starts are simulated but not summed). ``eps`` has shape
    (n_paths, len(future_dates)); passing the same draws to every variant gives
    common random numbers across variants.
    """
    if future_dates[0] != model.origin:
        raise ValueError("Simulation must start on the day after the last observation.")
    n_paths, n_days = eps.shape
    if n_days != len(future_dates) or season_mask.size != n_days:
        raise ValueError("eps, future_dates and season_mask must have matching lengths.")

    mean = model.seasonal_mean(future_dates)
    sig = model.sigma(future_dates)
    p = model.phi.size
    state = np.tile(model.last_anomalies, (n_paths, 1))  # columns: lag 1, lag 2, ...
    h = {b: np.zeros(n_paths) for b in bases}
    lin = {b: np.zeros(n_paths) for b in bases}
    for j in range(n_days):
        x_new = state @ model.phi + sig[j] * eps[:, j]
        if season_mask[j]:
            temp = mean[j] + x_new
            for b in bases:
                gap = b - temp
                lin[b] += gap
                h[b] += np.maximum(gap, 0.0)
        if p > 1:
            state[:, 1:] = state[:, :-1]
        state[:, 0] = x_new
    return {b: SimulatedIndex(h[b], lin[b]) for b in bases}


@dataclass(frozen=True)
class GaussianMoments:
    """Exact moments under the model (no Monte Carlo)."""

    day_mean_temp: np.ndarray  # E[T_t] for every future day
    day_var_temp: np.ndarray  # Var[T_t]
    sum_anomaly_mean: float  # E[sum over season days of X_t]
    sum_anomaly_var: float  # Var[sum over season days of X_t]

    def linear_index(self, base: float, season_mask: np.ndarray) -> tuple[float, float]:
        """Mean and sd of L = sum over season days of (base - T_t), which is Gaussian."""
        mean_l = float(np.sum(base - self.day_mean_temp[season_mask]))
        return mean_l, float(np.sqrt(self.sum_anomaly_var))

    def expected_index(self, base: float, season_mask: np.ndarray) -> float:
        """Exact E[H]: sum over season days of E[max(base - T_t, 0)] with T_t Gaussian."""
        return float(
            np.sum(
                expected_hdd_gaussian(
                    base, self.day_mean_temp[season_mask], self.day_var_temp[season_mask]
                )
            )
        )


def gaussian_moments(
    model: DailyModel, future_dates: pd.DatetimeIndex, season_mask: np.ndarray
) -> GaussianMoments:
    """Propagate the mean and covariance of the augmented state (X_t, ..., X_{t-p+1}, S_t).

    S_t is the running sum of X over season days. With z_t = A z_{t-1} + b_t eps_t and
    S_t = S_{t-1} + m_t X_t, the augmented transition is linear, so
    mu_t = F_t mu_{t-1} and P_t = F_t P_{t-1} F_t' + g_t g_t', starting from the
    observed anomalies (zero covariance).
    """
    p = model.phi.size
    a = companion(model.phi)
    sig = model.sigma(future_dates)
    mean_seasonal = model.seasonal_mean(future_dates)
    n = len(future_dates)

    mu = np.concatenate([model.last_anomalies, [0.0]])
    cov = np.zeros((p + 1, p + 1))
    day_mean = np.empty(n)
    day_var = np.empty(n)
    for j in range(n):
        m = 1.0 if season_mask[j] else 0.0
        f = np.eye(p + 1)
        f[:p, :p] = a
        f[p, :p] = m * a[0, :]
        g = np.zeros(p + 1)
        g[0] = sig[j]
        g[p] = m * sig[j]
        mu = f @ mu
        cov = f @ cov @ f.T + np.outer(g, g)
        day_mean[j] = mu[0]
        day_var[j] = cov[0, 0]
    return GaussianMoments(
        day_mean_temp=mean_seasonal + day_mean,
        day_var_temp=day_var,
        sum_anomaly_mean=float(mu[p]),
        sum_anomaly_var=float(cov[p, p]),
    )
