"""Parse and validate the HadCET daily mean temperature file.

The Met Office file has one row per day: an ISO date and a temperature to 0.1 degC.
Missing values are coded -99.9. Any header or footer lines are skipped and counted.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

MISSING_CODE = -99.9
_ROW = re.compile(r"^\s*(\d{4}-\d{2}-\d{2})\s+(-?\d+(?:\.\d+)?)\s*$")


@dataclass(frozen=True)
class ParseReport:
    n_rows: int
    n_skipped_lines: int
    n_missing_values: int
    first_date: pd.Timestamp
    last_date: pd.Timestamp


def parse_daily_text(text: str) -> tuple[pd.Series, ParseReport]:
    """Parse the daily file into a Series of degC indexed by date.

    Raises ValueError if no rows parse, dates repeat, or dates are out of order.
    """
    dates: list[str] = []
    values: list[float] = []
    skipped = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        match = _ROW.match(line)
        if match is None:
            skipped += 1
            continue
        dates.append(match.group(1))
        values.append(float(match.group(2)))
    if not dates:
        raise ValueError("No daily rows parsed: the file format may have changed.")

    index = pd.DatetimeIndex(pd.to_datetime(dates, format="%Y-%m-%d"), name="date")
    series = pd.Series(values, index=index, name="temp_c", dtype=float)
    if index.has_duplicates:
        raise ValueError("Duplicate dates in the daily file.")
    if not index.is_monotonic_increasing:
        raise ValueError("Dates in the daily file are not in increasing order.")
    missing = np.isclose(series.to_numpy(), MISSING_CODE)
    series[missing] = np.nan
    report = ParseReport(
        n_rows=len(series),
        n_skipped_lines=skipped,
        n_missing_values=int(missing.sum()),
        first_date=index[0],
        last_date=index[-1],
    )
    return series, report


def load_daily(path: Path) -> tuple[pd.Series, ParseReport]:
    """Read and parse the raw daily file."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` first."
        )
    return parse_daily_text(path.read_text(encoding="utf-8", errors="replace"))


def restrict_and_validate(series: pd.Series, start: str) -> pd.Series:
    """Keep dates from ``start`` on and check the series is gap-free and plausible.

    Every calendar day from ``start`` to the last date must be present and non-missing,
    and values must lie in [-30, 40] degC. Raises ValueError otherwise.
    """
    kept = series.loc[pd.Timestamp(start):]
    if kept.empty:
        raise ValueError(f"No data on or after {start}.")
    full = pd.date_range(kept.index[0], kept.index[-1], freq="D")
    if kept.index[0] != pd.Timestamp(start):
        raise ValueError(f"Series starts {kept.index[0].date()}, expected {start}.")
    gaps = full.difference(kept.index)
    if len(gaps) > 0:
        raise ValueError(f"{len(gaps)} calendar days missing, first {gaps[0].date()}.")
    if kept.isna().any():
        first_nan = kept.index[kept.isna()][0]
        raise ValueError(f"{int(kept.isna().sum())} missing values, first {first_nan.date()}.")
    out_of_range = (kept < -30.0) | (kept > 40.0)
    if out_of_range.any():
        raise ValueError(f"Implausible values on {list(kept.index[out_of_range].date)[:5]}.")
    return kept


def slice_hash(series: pd.Series, end: str) -> str:
    """SHA-256 of the data used, from the series start to ``end``, in a canonical text form.

    The live Met Office file gains a row every day, so a hash of the whole file changes
    daily. This hash changes only if the values actually used change.
    """
    used = series.loc[: pd.Timestamp(end)]
    lines = [f"{d:%Y-%m-%d},{v:.1f}" for d, v in zip(used.index, used.to_numpy())]
    return hashlib.sha256("\n".join(lines).encode("ascii")).hexdigest()
