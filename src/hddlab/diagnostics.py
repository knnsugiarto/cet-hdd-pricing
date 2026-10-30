"""Data and model diagnostics that are reported but do not drive any modelling choice."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox

from hddlab.daily_model import DailySpec, fit_daily_model
from hddlab.seasons import SeasonSpec, realised_index


def complete_seasons(temps: pd.Series, spec: SeasonSpec) -> list[int]:
    """Season years whose every day is in the data."""
    first = temps.index[0]
    last = temps.index[-1]
    years = range(first.year - 1, last.year + 1)
    return [y for y in years if spec.start_date(y) >= first and spec.end_date(y) <= last]


def seasonal_index_series(
    temps: pd.Series, spec: SeasonSpec, base: float
) -> pd.Series:
    """Realised H for every complete season."""
    years = complete_seasons(temps, spec)
    return pd.Series(
        [realised_index(temps, y, spec, base) for y in years], index=years, name="H"
    )


def warm_day_counts(
    temps: pd.Series, spec: SeasonSpec, bases: tuple[float, ...]
) -> pd.DataFrame:
    """Days in complete seasons with T at or above each base temperature, by month.

    When this is near zero, H = sum(base - T) almost surely and the Gaussian
    closed form for the linear index is a good approximation for H.
    """
    years = complete_seasons(temps, spec)
    days = pd.DatetimeIndex(np.concatenate([spec.dates(y).to_numpy() for y in years]))
    values = temps.reindex(days)
    months = days.month
    order = sorted(set(months), key=lambda m: (m < spec.start[0], m))
    table = {
        f"T >= {b:g} C": [int(((values >= b) & (months == m)).sum()) for m in order]
        for b in bases
    }
    table["days"] = [int((months == m).sum()) for m in order]
    month_names = [pd.Timestamp(2000, m, 1).strftime("%b") for m in order]
    return pd.DataFrame(table, index=month_names)


def ljung_box_table(
    temps_before: pd.Series, specs: list[DailySpec], lags: int = 10
) -> pd.DataFrame:
    """Ljung-Box test on AR residuals for each spec, fitted on the given data.

    With about 10,000 daily residuals the test rejects even tiny autocorrelation, so
    read it with the residual autocorrelations themselves, not as a pass/fail gate.
    """
    rows = []
    for spec in specs:
        model = fit_daily_model(temps_before, spec)
        lb = acorr_ljungbox(model.residuals, lags=[lags], return_df=True)
        rows.append(
            {
                "variant": spec.label,
                "n_residuals": model.residuals.size,
                "phi": ", ".join(f"{x:.3f}" for x in model.phi),
                "acf1": model.resid_acf[0],
                "acf2": model.resid_acf[1],
                "acf3": model.resid_acf[2],
                f"lb_stat_{lags}": float(lb["lb_stat"].iloc[0]),
                f"lb_pvalue_{lags}": float(lb["lb_pvalue"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)
