"""The core cross-check: Monte Carlo prices from the daily model against exact results.

L = sum(base - T_t) is exactly Gaussian under the model, so a call on L has a Bachelier
closed form. H = sum max(base - T_t, 0) >= L path by path, so the Monte Carlo price of
the call on H can only exceed the call on L; E[H] is known exactly day by day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hddlab.daily_model import DailySpec, fit_daily_model, gaussian_moments, simulate_index
from hddlab.distributions import bachelier_call
from hddlab.seasons import SeasonSpec

N_PATHS = 200_000


@pytest.fixture(scope="module")
def setup(temps: pd.Series):
    spec = SeasonSpec((11, 1), (3, 31))
    before = temps.loc[:"2015-10-31"]
    model = fit_daily_model(before, DailySpec(30, 3, 3, "seasonal"))
    future = spec.dates(2015)
    mask = np.ones(len(future), dtype=bool)
    eps = np.random.default_rng(11).standard_normal((N_PATHS, len(future)))
    sims = simulate_index(model, future, mask, (18.0, 15.5), eps)
    moments = gaussian_moments(model, future, mask)
    return sims, moments, mask


@pytest.mark.parametrize("base", [18.0, 15.5])
@pytest.mark.parametrize("offset_sd", [-1.0, 0.0, 1.0])
def test_call_on_linear_index_matches_bachelier(setup, base: float, offset_sd: float) -> None:
    sims, moments, mask = setup
    mean_l, sd_l = moments.linear_index(base, mask)
    strike = mean_l + offset_sd * sd_l
    payoff = np.maximum(sims[base].l_samples - strike, 0.0)
    se = payoff.std(ddof=1) / np.sqrt(N_PATHS)
    assert abs(payoff.mean() - bachelier_call(mean_l, sd_l, strike)) < 4 * se


@pytest.mark.parametrize("base", [18.0, 15.5])
def test_mean_of_h_matches_exact_expectation(setup, base: float) -> None:
    sims, moments, mask = setup
    h = sims[base].h_samples
    assert abs(h.mean() - moments.expected_index(base, mask)) < 4 * h.std(ddof=1) / np.sqrt(N_PATHS)


@pytest.mark.parametrize("base", [18.0, 15.5])
def test_h_dominates_l_pathwise(setup, base: float) -> None:
    sims, _, _ = setup
    assert np.all(sims[base].h_samples >= sims[base].l_samples - 1e-9)


def test_truncation_matters_more_at_lower_base(setup) -> None:
    sims, _, _ = setup
    gap = {b: np.mean(sims[b].h_samples - sims[b].l_samples) for b in (18.0, 15.5)}
    assert gap[15.5] > gap[18.0] >= 0.0
