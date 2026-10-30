"""HDD arithmetic, season calendars and calendar adjustment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hddlab.seasons import SeasonSpec, calendar_adjusted_totals, daily_hdd, realised_index

NOV_MAR = SeasonSpec((11, 1), (3, 31))


def test_daily_hdd_hand_example() -> None:
    # 18 - 20 < 0 -> 0; exactly at base -> 0; 0.5 below; 20 below.
    assert np.allclose(daily_hdd(np.array([20.0, 18.0, 17.5, -2.0]), 18.0), [0.0, 0.0, 0.5, 20.0])


def test_seasonal_total_hand_example() -> None:
    # A 151-day season at a constant 8 degC: 10 HDD a day.
    dates = NOV_MAR.dates(2022)
    temps = pd.Series(8.0, index=dates)
    assert realised_index(temps, 2022, NOV_MAR, 18.0) == pytest.approx(1510.0)


@pytest.mark.parametrize(
    ("year", "length", "has_feb29"),
    [
        (2022, 151, False),  # Nov 2022 - Mar 2023
        (2023, 152, True),  # Nov 2023 - Mar 2024 (2024 is a leap year)
        (1899, 151, False),  # 1900 is not a leap year (century rule)
        (1999, 152, True),  # 2000 is a leap year (400 rule)
    ],
)
def test_season_length_and_leap_day(year: int, length: int, has_feb29: bool) -> None:
    assert NOV_MAR.length(year) == length
    assert NOV_MAR.includes_feb29(year) is has_feb29


def test_season_must_cross_year_end() -> None:
    with pytest.raises(ValueError):
        SeasonSpec((1, 1), (3, 31))


def test_calendar_adjustment_drops_and_adds_feb29() -> None:
    dates = pd.date_range("2022-11-01", "2024-03-31", freq="D")
    temps = pd.Series(10.0, index=dates)
    temps[pd.Timestamp("2023-02-28")] = 4.0
    temps[pd.Timestamp("2023-03-01")] = 6.0
    # Season 2022 has no 29 Feb; season 2023 has one.
    as_non_leap = calendar_adjusted_totals(temps, [2022, 2023], NOV_MAR, 18.0, target_has_feb29=False)
    as_leap = calendar_adjusted_totals(temps, [2022, 2023], NOV_MAR, 18.0, target_has_feb29=True)
    base_2022 = 8.0 * 149 + 14.0 + 12.0
    assert as_non_leap[0] == pytest.approx(base_2022)
    assert as_non_leap[1] == pytest.approx(8.0 * 151)  # 29 Feb 2024 dropped
    # Proxy 29 Feb for 2022: HDD of the mean of 4 and 6 degC = 13.
    assert as_leap[0] == pytest.approx(base_2022 + 13.0)
    assert as_leap[1] == pytest.approx(8.0 * 152)


def test_incomplete_season_raises() -> None:
    temps = pd.Series(5.0, index=pd.date_range("2022-11-01", "2023-03-30", freq="D"))
    with pytest.raises(ValueError):
        realised_index(temps, 2022, NOV_MAR, 18.0)
