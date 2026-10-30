"""Parsing and validation of the Met Office daily file."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hddlab.data import parse_daily_text, restrict_and_validate, slice_hash
from hddlab.synthetic import synthetic_cet, to_hadcet_text

SAMPLE = """Date        Value
 1900-10-31   9.1
 1900-11-01   8.5
 1900-11-02 -99.9
 1900-11-03  -1.2
"""


def test_parse_sample_with_header_and_missing() -> None:
    series, report = parse_daily_text(SAMPLE)
    assert report.n_rows == 4
    assert report.n_skipped_lines == 1
    assert report.n_missing_values == 1
    assert np.isnan(series[pd.Timestamp("1900-11-02")])
    assert series[pd.Timestamp("1900-11-03")] == pytest.approx(-1.2)


def test_validation_rejects_missing_values() -> None:
    series, _ = parse_daily_text(SAMPLE)
    with pytest.raises(ValueError, match="missing"):
        restrict_and_validate(series, "1900-11-01")


def test_validation_rejects_gaps() -> None:
    series = pd.Series(
        [1.0, 2.0, 3.0], index=pd.to_datetime(["1900-11-01", "1900-11-02", "1900-11-04"])
    )
    with pytest.raises(ValueError, match="missing"):
        restrict_and_validate(series, "1900-11-01")


def test_parse_rejects_non_data() -> None:
    with pytest.raises(ValueError):
        parse_daily_text("not a data file\n")


def test_round_trip_and_stable_hash() -> None:
    original = synthetic_cet(start="1990-01-01", end="1991-12-31")
    parsed, report = parse_daily_text(to_hadcet_text(original))
    assert report.n_rows == len(original)
    pd.testing.assert_series_equal(parsed, original, check_names=False, check_freq=False)
    # Appending later days must not change the hash of the slice used.
    longer = synthetic_cet(start="1990-01-01", end="1992-06-30")
    assert slice_hash(parsed, "1991-06-30") == slice_hash(longer, "1991-06-30")
