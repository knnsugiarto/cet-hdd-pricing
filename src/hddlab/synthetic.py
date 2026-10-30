"""Synthetic CET-like daily temperatures, for tests and the ``--synthetic`` smoke run.

Nothing here is used when real data are present. The numbers are chosen to look like
CET (annual mean about 10 degC, coldest in mid-January, winter day-to-day sd about
2 degC) so the pipeline exercises realistic magnitudes; they are not estimates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hddlab.data import MISSING_CODE


def synthetic_cet(
    start: str = "1880-01-01",
    end: str = "2026-10-27",
    seed: int = 7,
    trend_per_year: float = 0.01,
    phi: float = 0.78,
    slow_innovation_sd: float = 0.07,
) -> pd.Series:
    """Daily temperatures: seasonal cycle + linear trend + AR(1) + a slow AR(1) component.

    The slow component (daily persistence 0.995, stationary sd about 0.7 degC) adds
    year-to-year variability that a short-memory AR fit does not capture, as long
    memory does in real temperatures.
    """
    dates = pd.date_range(start, end, freq="D", name="date")
    n = len(dates)
    t = (dates - pd.Timestamp("1950-01-01")).days.to_numpy() / 365.2425
    phase = 2.0 * np.pi * t
    seasonal = 10.0 + trend_per_year * t - 6.8 * np.cos(phase - 2.0 * np.pi * 15 / 365.25)
    sigma = 2.0 + 0.4 * np.cos(phase - 2.0 * np.pi * 15 / 365.25)
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal((n, 2))  # row-wise, so a longer series extends a shorter one
    eps, eps_slow = shocks[:, 0], shocks[:, 1]
    fast = np.zeros(n)
    slow = np.zeros(n)
    for i in range(1, n):
        fast[i] = phi * fast[i - 1] + sigma[i] * eps[i]
        slow[i] = 0.995 * slow[i - 1] + slow_innovation_sd * eps_slow[i]
    values = np.round(seasonal + fast + slow, 1)
    return pd.Series(values, index=dates, name="temp_c")


def to_hadcet_text(series: pd.Series, header: bool = True) -> str:
    """Render a series in the Met Office daily file layout (date, value to 0.1 degC)."""
    lines = ["Date  Value"] if header else []
    for date, value in zip(series.index, series.to_numpy()):
        v = MISSING_CODE if np.isnan(value) else value
        lines.append(f" {date:%Y-%m-%d} {v:5.1f}")
    return "\n".join(lines) + "\n"
