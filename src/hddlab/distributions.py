"""Predictive distributions of the seasonal index H, and option prices under them.

Each method returns a ``Forecast``: either a Gaussian (index method) or an ensemble of
equally weighted values (burn analysis: past seasons; daily model: Monte Carlo paths).
Prices are undiscounted expected payoffs per unit tick (decision D10).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.special import ndtr, ndtri

_SQRT_PI = np.sqrt(np.pi)


def _pdf(z: np.ndarray | float) -> np.ndarray | float:
    return np.exp(-0.5 * np.square(z)) / np.sqrt(2.0 * np.pi)


# ---------------------------------------------------------------------------
# Gaussian closed forms
# ---------------------------------------------------------------------------


def bachelier_call(mean: float, sd: float, strike: float) -> float:
    """E[max(H - K, 0)] for H ~ N(mean, sd^2): (m - K) Phi(d) + s phi(d), d = (m - K)/s."""
    d = (mean - strike) / sd
    return float((mean - strike) * ndtr(d) + sd * _pdf(d))


def bachelier_put(mean: float, sd: float, strike: float) -> float:
    """E[max(K - H, 0)] for H ~ N(mean, sd^2)."""
    d = (mean - strike) / sd
    return float((strike - mean) * ndtr(-d) + sd * _pdf(d))


def call_second_moment(mean: float, sd: float, strike: float) -> float:
    """E[max(H - K, 0)^2] for H ~ N(mean, sd^2)."""
    a = mean - strike
    d = a / sd
    return float((a * a + sd * sd) * ndtr(d) + a * sd * _pdf(d))


def put_second_moment(mean: float, sd: float, strike: float) -> float:
    """E[max(K - H, 0)^2] for H ~ N(mean, sd^2)."""
    a = strike - mean
    d = a / sd
    return float((a * a + sd * sd) * ndtr(d) + a * sd * _pdf(d))


def expected_hdd_gaussian(base: float, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    """E[max(base - T, 0)] for T ~ N(mean, var), elementwise."""
    sd = np.sqrt(var)
    z = (base - mean) / sd
    return (base - mean) * ndtr(z) + sd * _pdf(z)


def gaussian_crps(mean: float, sd: float, y: float) -> float:
    """Continuous ranked probability score of N(mean, sd^2) at outcome y (lower is better)."""
    z = (y - mean) / sd
    return float(sd * (z * (2.0 * ndtr(z) - 1.0) + 2.0 * _pdf(z) - 1.0 / _SQRT_PI))


def ensemble_crps(sorted_values: np.ndarray, y: float) -> float:
    """CRPS of the empirical distribution of ``sorted_values`` at y.

    Uses CRPS = E|X - y| - 0.5 E|X - X'| with the sorted-sample identity
    E|X - X'| = (2 / n^2) sum_i (2i - n - 1) x_(i), which is exact for the
    empirical distribution (the one the burn method prices with).
    """
    x = sorted_values
    n = x.size
    i = np.arange(1, n + 1)
    term1 = np.mean(np.abs(x - y))
    half_mean_abs_diff = np.sum((2 * i - n - 1) * x) / (n * n)
    return float(term1 - half_mean_abs_diff)


# ---------------------------------------------------------------------------
# Forecast container
# ---------------------------------------------------------------------------


@dataclass
class Forecast:
    """Predictive distribution of the seasonal index H for one season and one method."""

    kind: Literal["gaussian", "ensemble"]
    mean: float
    sd: float
    samples: np.ndarray | None = None  # sorted ascending, ensemble only
    extras: dict[str, float] = field(default_factory=dict)

    @classmethod
    def gaussian(cls, mean: float, sd: float) -> "Forecast":
        if not sd > 0:
            raise ValueError("Gaussian forecast needs a positive sd.")
        return cls("gaussian", float(mean), float(sd))

    @classmethod
    def ensemble(cls, values: np.ndarray, extras: dict[str, float] | None = None) -> "Forecast":
        values = np.sort(np.asarray(values, dtype=float))
        if values.size < 2:
            raise ValueError("Ensemble forecast needs at least two values.")
        return cls(
            "ensemble",
            float(values.mean()),
            float(values.std(ddof=0)),  # sd of the empirical distribution itself
            values,
            dict(extras or {}),
        )

    # -- distribution functions ------------------------------------------------

    def cdf(self, y: float) -> float:
        """P(H <= y). For ensembles, ties count half (mid-rank PIT)."""
        if self.kind == "gaussian":
            return float(ndtr((y - self.mean) / self.sd))
        assert self.samples is not None
        below = np.searchsorted(self.samples, y, side="left")
        at_or_below = np.searchsorted(self.samples, y, side="right")
        return float((below + at_or_below) / (2.0 * self.samples.size))

    def quantile(self, q: float) -> float:
        if self.kind == "gaussian":
            return float(self.mean + self.sd * ndtri(q))
        assert self.samples is not None
        return float(np.quantile(self.samples, q))

    def interval(self, level: float) -> tuple[float, float]:
        """Central prediction interval with the given coverage."""
        tail = (1.0 - level) / 2.0
        return self.quantile(tail), self.quantile(1.0 - tail)

    def crps(self, y: float) -> float:
        if self.kind == "gaussian":
            return gaussian_crps(self.mean, self.sd, y)
        assert self.samples is not None
        return ensemble_crps(self.samples, y)

    # -- pricing (per unit tick, undiscounted) -----------------------------------

    def option_moments(self, kind: Literal["call", "put"], strike: float) -> tuple[float, float]:
        """Expected payoff and payoff sd of max(H - K, 0) (call) or max(K - H, 0) (put)."""
        if self.kind == "gaussian":
            if kind == "call":
                m1 = bachelier_call(self.mean, self.sd, strike)
                m2 = call_second_moment(self.mean, self.sd, strike)
            else:
                m1 = bachelier_put(self.mean, self.sd, strike)
                m2 = put_second_moment(self.mean, self.sd, strike)
            return m1, float(np.sqrt(max(m2 - m1 * m1, 0.0)))
        assert self.samples is not None
        sign = 1.0 if kind == "call" else -1.0
        payoff = np.maximum(sign * (self.samples - strike), 0.0)
        return float(payoff.mean()), float(payoff.std(ddof=0))
