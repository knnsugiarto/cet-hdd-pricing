"""Season calendars, daily heating degree days and seasonal HDD totals.

A season is labelled by the year it starts: season 1950 runs from 1 Nov 1950 to
31 Mar 1951 for the default contract. A season includes 29 February when its
second calendar year is a leap year, so it has 151 or 152 days.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SeasonSpec:
    """A season that starts in year y and ends in year y + 1."""

    start: tuple[int, int]  # (month, day) in year y
    end: tuple[int, int]  # (month, day) in year y + 1

    def __post_init__(self) -> None:
        if self.start[0] <= self.end[0]:
            raise ValueError("A season must cross the year end (start month > end month).")

    def start_date(self, year: int) -> pd.Timestamp:
        return pd.Timestamp(year=year, month=self.start[0], day=self.start[1])

    def end_date(self, year: int) -> pd.Timestamp:
        return pd.Timestamp(year=year + 1, month=self.end[0], day=self.end[1])

    def dates(self, year: int) -> pd.DatetimeIndex:
        return pd.date_range(self.start_date(year), self.end_date(year), freq="D")

    def includes_feb29(self, year: int) -> bool:
        """True if the season spans 29 February of year + 1 and that day exists."""
        spans_end_feb = self.end[0] >= 3 or (self.end[0] == 2 and self.end[1] == 29)
        return spans_end_feb and pd.Timestamp(year=year + 1, month=1, day=1).is_leap_year

    def length(self, year: int) -> int:
        return len(self.dates(year))


def daily_hdd(temps: np.ndarray | pd.Series, base: float) -> np.ndarray:
    """Heating degree days: max(base - T, 0) for each day."""
    return np.maximum(base - np.asarray(temps, dtype=float), 0.0)


def realised_index(temps: pd.Series, year: int, spec: SeasonSpec, base: float) -> float:
    """Seasonal HDD total on the season's own calendar. Raises if any day is missing."""
    dates = spec.dates(year)
    values = temps.reindex(dates)
    if values.isna().any():
        raise ValueError(f"Season {year} is incomplete in the data.")
    return float(daily_hdd(values.to_numpy(), base).sum())


def calendar_adjusted_totals(
    temps: pd.Series,
    years: list[int] | np.ndarray,
    spec: SeasonSpec,
    base: float,
    target_has_feb29: bool,
) -> np.ndarray:
    """Seasonal HDD totals of past seasons, put on the target season's calendar.

    A past leap season loses 29 Feb when the target has none. A past non-leap season
    gains a proxy 29 Feb when the target has one: the HDD of the mean of 28 Feb and
    1 Mar temperatures (decision D8). This keeps a one-day calendar difference
    (about 0.1 sd of the index) out of the burn and index methods.
    """
    totals = np.empty(len(years), dtype=float)
    for i, year in enumerate(years):
        dates = spec.dates(int(year))
        values = temps.reindex(dates)
        if values.isna().any():
            raise ValueError(f"Season {year} is incomplete in the data.")
        has_feb29 = spec.includes_feb29(int(year))
        if has_feb29 and not target_has_feb29:
            values = values.drop(pd.Timestamp(year=int(year) + 1, month=2, day=29))
        hdd_sum = float(daily_hdd(values.to_numpy(), base).sum())
        if target_has_feb29 and not has_feb29:
            t28 = temps.loc[pd.Timestamp(year=int(year) + 1, month=2, day=28)]
            t01 = temps.loc[pd.Timestamp(year=int(year) + 1, month=3, day=1)]
            hdd_sum += float(daily_hdd(np.array([(t28 + t01) / 2.0]), base)[0])
        totals[i] = hdd_sum
    return totals
