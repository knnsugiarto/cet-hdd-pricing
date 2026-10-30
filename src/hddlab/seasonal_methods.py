"""Burn analysis and index modelling: methods that use only past seasonal totals.

Both take the totals of the previous ``window`` seasons (already on the target
season's calendar) and optionally remove a linear trend in the season year.
They differ only in the shape of the predictive distribution:

* burn analysis uses the (detrended) past totals themselves, an empirical distribution;
* index modelling fits a normal distribution to them and prices in closed form.
"""

from __future__ import annotations

import numpy as np

from hddlab.distributions import Forecast


def trend_adjusted(
    totals: np.ndarray, years: np.ndarray, target_year: int, trend: str
) -> np.ndarray:
    """Past totals moved to the target year's trend level.

    ``trend="none"`` returns the totals unchanged. ``trend="linear"`` fits
    H_j = a + b j by least squares and returns H_j + b (target_year - j): the fitted
    value at the target year plus each season's residual.
    """
    totals = np.asarray(totals, dtype=float)
    if trend == "none":
        return totals.copy()
    if trend == "linear":
        slope, _ = np.polyfit(np.asarray(years, dtype=float), totals, 1)
        return totals + slope * (target_year - np.asarray(years, dtype=float))
    raise ValueError(f"Unknown trend form {trend!r}.")


def burn_forecast(
    totals: np.ndarray, years: np.ndarray, target_year: int, trend: str
) -> Forecast:
    """Burn analysis: the empirical distribution of the (trend-adjusted) past totals."""
    return Forecast.ensemble(trend_adjusted(totals, years, target_year, trend))


def index_forecast(
    totals: np.ndarray, years: np.ndarray, target_year: int, trend: str
) -> Forecast:
    """Index modelling: a normal predictive distribution for the target season's total.

    The predictive sd includes the estimation error of the mean (or of the trend line
    at the target year), not just the residual scatter (decision D12):

    * no trend: sd = s * sqrt(1 + 1/n), s the sample sd;
    * linear:   sd = s * sqrt(1 + 1/n + (y0 - ybar)^2 / Sxx), s the residual sd (n - 2 dof).
    """
    totals = np.asarray(totals, dtype=float)
    years = np.asarray(years, dtype=float)
    n = totals.size
    if trend == "none":
        mean = totals.mean()
        s = totals.std(ddof=1)
        sd = s * np.sqrt(1.0 + 1.0 / n)
    elif trend == "linear":
        slope, intercept = np.polyfit(years, totals, 1)
        resid = totals - (intercept + slope * years)
        s = np.sqrt(np.sum(resid**2) / (n - 2))
        ybar = years.mean()
        sxx = np.sum((years - ybar) ** 2)
        mean = intercept + slope * target_year
        sd = s * np.sqrt(1.0 + 1.0 / n + (target_year - ybar) ** 2 / sxx)
    else:
        raise ValueError(f"Unknown trend form {trend!r}.")
    return Forecast.gaussian(mean, sd)


def reference_strikes(
    raw_totals: np.ndarray, sd_multiples: tuple[float, ...]
) -> dict[float, float]:
    """Strikes from the mean and sd of past raw totals: {multiple: mean + multiple * sd}.

    The same strikes are used for every method in a season (decision D9), so the
    methods price the same contracts.
    """
    m = float(np.mean(raw_totals))
    s = float(np.std(raw_totals, ddof=1))
    return {k: m + k * s for k in sd_multiples}
