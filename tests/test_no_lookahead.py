"""Forecasts must not change when anything after the pricing cutoff changes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hddlab.walkforward import build_variants, contract_season, forecast_rows, forecast_season, run_walkforward


@pytest.mark.parametrize("year", [1960, 2010])
def test_forecasts_ignore_data_after_cutoff(small_cfg, temps: pd.Series, year: int) -> None:
    spec = contract_season(small_cfg)
    cutoff = spec.start_date(year)
    variants = build_variants(small_cfg)

    altered = temps.copy()
    after = altered.index >= cutoff
    rng = np.random.default_rng(0)
    altered[after] = rng.uniform(-20.0, 30.0, size=int(after.sum()))  # scramble the future

    def rows(series: pd.Series) -> pd.DataFrame:
        before = series.loc[: cutoff - pd.Timedelta(days=1)]
        sf = forecast_season(before, year, variants, small_cfg, spec, small_cfg.all_bases)
        return pd.DataFrame(forecast_rows(sf, small_cfg))

    pd.testing.assert_frame_equal(rows(temps), rows(altered))


def test_harness_truncates_before_forecasting(small_cfg, temps: pd.Series) -> None:
    """End to end: scrambling data after the last evaluated season's start changes only outcomes."""
    variants = [v for v in build_variants(small_cfg) if v.label.endswith("window=30")][:3]
    year = 1955
    cutoff = contract_season(small_cfg).start_date(year)
    altered = temps.copy()
    altered[altered.index >= cutoff] += 3.0
    a = run_walkforward(temps, [year], variants, small_cfg, (18.0,), "dev")
    b = run_walkforward(altered, [year], variants, small_cfg, (18.0,), "dev")
    forecast_cols = ["mean", "sd", "lo90", "hi90", "price_call_atm", "strike_call_atm"]
    pd.testing.assert_frame_equal(a[forecast_cols], b[forecast_cols])
    assert not np.allclose(a["realised"], b["realised"])


def test_forecast_rejects_data_inside_the_season(small_cfg, temps: pd.Series) -> None:
    spec = contract_season(small_cfg)
    with pytest.raises(ValueError):
        forecast_season(temps.loc[:"1960-11-05"], 1960, build_variants(small_cfg)[:1], small_cfg, spec, (18.0,))
