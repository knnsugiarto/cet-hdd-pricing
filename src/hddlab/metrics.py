"""Evaluation statistics across seasons.

Season-level errors can be serially correlated (a trend the model misses persists for
years), so standard errors of means use Newey-West with Bartlett weights (decision D14).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

# z_{0.975} + z_{0.80}: a true bias this many standard errors from zero is detected
# by a two-sided 5% test with 80% probability.
MDE_MULTIPLIER = float(norm.ppf(0.975) + norm.ppf(0.80))


def newey_west_se(x: np.ndarray, lags: int) -> float:
    """Standard error of the mean of ``x`` allowing autocorrelation up to ``lags``."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 2:
        return float("nan")
    d = x - x.mean()
    long_run_var = np.dot(d, d) / n
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1.0 - lag / (lags + 1.0)
        long_run_var += 2.0 * weight * np.dot(d[:-lag], d[lag:]) / n
    return float(np.sqrt(max(long_run_var, 0.0) / n))


@dataclass(frozen=True)
class MeanEstimate:
    """A sample mean with its standard error and the smallest bias it could detect."""

    mean: float
    se: float
    n: int

    @property
    def t(self) -> float:
        return self.mean / self.se if self.se > 0 else float("nan")

    @property
    def mde(self) -> float:
        """Minimum detectable effect: 5% two-sided test, 80% power."""
        return MDE_MULTIPLIER * self.se


def mean_estimate(x: np.ndarray, lags: int) -> MeanEstimate:
    x = np.asarray(x, dtype=float)
    return MeanEstimate(float(x.mean()), newey_west_se(x, lags), int(x.size))


@dataclass(frozen=True)
class CoverageEstimate:
    level: float
    rate: float
    n: int

    @property
    def se(self) -> float:
        """Binomial SE under the nominal coverage."""
        return float(np.sqrt(self.level * (1.0 - self.level) / self.n))

    @property
    def z(self) -> float:
        return (self.rate - self.level) / self.se


def coverage(hits: np.ndarray, level: float) -> CoverageEstimate:
    hits = np.asarray(hits, dtype=bool)
    return CoverageEstimate(level, float(hits.mean()), int(hits.size))


def standardised_rms(z_scores: np.ndarray) -> tuple[float, float]:
    """Root mean square of standardised errors z = (y - mean) / sd, and its approximate SE.

    Near 1 when a forecast's mean and spread are both right; above 1 means the
    forecast is too narrow (or biased) for the outcomes it got.
    """
    z = np.asarray(z_scores, dtype=float)
    sd = float(np.sqrt(np.mean(z**2)))
    return sd, sd / np.sqrt(2.0 * z.size)


def pit_histogram(pit: np.ndarray, bins: int = 10) -> np.ndarray:
    """Counts of PIT values in equal bins on [0, 1]."""
    counts, _ = np.histogram(np.clip(pit, 0.0, 1.0), bins=bins, range=(0.0, 1.0))
    return counts
