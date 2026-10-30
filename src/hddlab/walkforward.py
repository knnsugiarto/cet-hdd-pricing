"""Walk-forward evaluation: price each season using only data available before it.

``forecast_season`` is the only function that sees temperatures, and it receives them
already truncated at the pricing cutoff. Realised outcomes are computed separately in
``score_forecasts``. The no-look-ahead test perturbs every value after the cutoff and
checks that ``forecast_season`` returns identical forecasts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from hddlab.config import Config
from hddlab.daily_model import (
    DailyModel,
    DailySpec,
    GaussianMoments,
    fit_daily_model,
    gaussian_moments,
    simulate_index,
)
from hddlab.distributions import Forecast, bachelier_call
from hddlab.seasonal_methods import burn_forecast, index_forecast, reference_strikes
from hddlab.seasons import SeasonSpec, calendar_adjusted_totals, realised_index

# (name, option type, strike as a multiple of the reference sd)
INSTRUMENTS: tuple[tuple[str, str, float], ...] = (
    ("call_lo", "call", -1.0),
    ("call_atm", "call", 0.0),
    ("call_hi", "call", 1.0),
    ("put_lo", "put", -1.0),
    ("put_atm", "put", 0.0),
    ("put_hi", "put", 1.0),
)


@dataclass(frozen=True)
class Variant:
    """One method configuration. ``label`` is its unique name in every results table."""

    family: str  # "burn", "index" or "daily"
    label: str
    window: int
    trend: str | None = None
    daily: DailySpec | None = None


def build_variants(cfg: Config) -> list[Variant]:
    """Every variant in the config grid, in a fixed order."""
    variants: list[Variant] = []
    for family, grid in (("burn", cfg.burn), ("index", cfg.index)):
        for trend in grid.trends:
            for window in grid.windows:
                label = f"{family}|trend={trend}|window={window}"
                variants.append(Variant(family, label, window, trend=trend))
    d = cfg.daily
    for p in d.ar_orders:
        for harmonics in d.harmonics:
            for vol in d.vol:
                for window in d.windows:
                    spec = DailySpec(window, harmonics, p, vol, d.vol_harmonics)
                    variants.append(Variant("daily", spec.label, window, daily=spec))
    return variants


def contract_season(cfg: Config) -> SeasonSpec:
    return SeasonSpec(cfg.contract.season_start, cfg.contract.season_end)


@dataclass
class SeasonForecast:
    """All methods' forecasts for one season, before the outcome is known."""

    year: int
    cutoff: pd.Timestamp
    strikes: dict[float, dict[float, float]]  # base -> {sd multiple: strike}
    forecasts: dict[tuple[str, float], Forecast] = field(default_factory=dict)  # (label, base)
    families: dict[str, str] = field(default_factory=dict)  # label -> family


def _past_seasons(year: int, spec: SeasonSpec, cutoff: pd.Timestamp, n: int) -> list[int]:
    years = list(range(year - n, year))
    if spec.end_date(years[-1]) >= cutoff:
        raise ValueError("A reference season is not complete before the pricing cutoff.")
    return years


def forecast_season(
    temps_before: pd.Series,
    year: int,
    variants: list[Variant],
    cfg: Config,
    spec: SeasonSpec,
    bases: tuple[float, ...],
) -> SeasonForecast:
    """Forecast season ``year`` from ``temps_before``, the data available at pricing.

    The pricing cutoff is the day after the last observation. It must not be later than
    the season start.
    """
    cutoff = temps_before.index[-1] + pd.Timedelta(days=1)
    if cutoff > spec.start_date(year):
        raise ValueError("Data run into the season: the pricing date must precede it.")
    target_leap = spec.includes_feb29(year)
    n_hist = max(
        [cfg.contract.strike_reference_seasons]
        + [v.window for v in variants if v.family in ("burn", "index")]
    )
    past = _past_seasons(year, spec, cutoff, n_hist)
    past_arr = np.array(past)
    totals = {
        b: calendar_adjusted_totals(temps_before, past, spec, b, target_leap) for b in bases
    }
    n_ref = cfg.contract.strike_reference_seasons
    strikes = {
        b: reference_strikes(totals[b][-n_ref:], cfg.contract.strike_sd_multiples) for b in bases
    }
    out = SeasonForecast(year, cutoff, strikes)

    for v in variants:
        if v.family not in ("burn", "index"):
            continue
        method = burn_forecast if v.family == "burn" else index_forecast
        for b in bases:
            out.forecasts[(v.label, b)] = method(
                totals[b][-v.window :], past_arr[-v.window :], year, v.trend
            )
        out.families[v.label] = v.family

    daily_variants = [v for v in variants if v.family == "daily"]
    if daily_variants:
        future = pd.date_range(cutoff, spec.end_date(year), freq="D")
        mask = np.asarray(future >= spec.start_date(year))
        rng = np.random.default_rng([cfg.daily.seed, year])
        eps = rng.standard_normal((cfg.daily.n_paths, len(future)))
        for v in daily_variants:
            assert v.daily is not None
            model = fit_daily_model(temps_before, v.daily, cfg.daily.acf_lags)
            sims = simulate_index(model, future, mask, bases, eps)
            moments = gaussian_moments(model, future, mask)
            for b in bases:
                out.forecasts[(v.label, b)] = _daily_forecast(
                    sims[b].h_samples, sims[b].l_samples, moments, mask, b, strikes[b][0.0], model
                )
            out.families[v.label] = "daily"
    return out


def _daily_forecast(
    h: np.ndarray,
    lin: np.ndarray,
    moments: GaussianMoments,
    mask: np.ndarray,
    base: float,
    atm_strike: float,
    model: DailyModel,
) -> Forecast:
    """Wrap Monte Carlo draws of H, with the closed-form checks stored as extras."""
    n = h.size
    mean_l, sd_l = moments.linear_index(base, mask)
    call_l = np.maximum(lin - atm_strike, 0.0)
    call_h = np.maximum(h - atm_strike, 0.0)
    se_call_l = call_l.std(ddof=1) / np.sqrt(n)
    exact_eh = moments.expected_index(base, mask)
    extras = {
        "cf_mean_l": mean_l,
        "cf_sd_l": sd_l,
        "z_mean_l": (lin.mean() - mean_l) / (lin.std(ddof=1) / np.sqrt(n)),
        "z_call_l": (call_l.mean() - bachelier_call(mean_l, sd_l, atm_strike)) / se_call_l,
        "z_mean_h": (h.mean() - exact_eh) / (h.std(ddof=1) / np.sqrt(n)),
        "exact_mean_h": exact_eh,
        "truncation_gap_atm": call_h.mean() - call_l.mean(),
        "mc_se_call_atm": call_h.std(ddof=1) / np.sqrt(n),
        "phi1": float(model.phi[0]),
        "resid_sd": model.resid_sd,
    }
    for k, r in enumerate(model.resid_acf, start=1):
        extras[f"resid_acf{k}"] = float(r)
    return Forecast.ensemble(h, extras)


def forecast_rows(sf: SeasonForecast, cfg: Config) -> list[dict]:
    """Scalar summary of every forecast in a season: moments, intervals, strikes, prices."""
    rows = []
    for (label, base), fc in sf.forecasts.items():
        row: dict = {
            "year": sf.year,
            "family": sf.families[label],
            "variant": label,
            "base": base,
            "mean": fc.mean,
            "sd": fc.sd,
        }
        for level in cfg.evaluation.intervals:
            lo, hi = fc.interval(level)
            row[f"lo{int(level * 100)}"] = lo
            row[f"hi{int(level * 100)}"] = hi
        for name, kind, mult in INSTRUMENTS:
            strike = sf.strikes[base][mult]
            price, payoff_sd = fc.option_moments(kind, strike)
            row[f"strike_{name}"] = strike
            row[f"price_{name}"] = price
            row[f"payoff_sd_{name}"] = payoff_sd
        row.update(fc.extras)
        rows.append(row)
    return rows


def score_forecasts(
    sf: SeasonForecast, rows: list[dict], temps_full: pd.Series, spec: SeasonSpec, cfg: Config
) -> list[dict]:
    """Add the realised index and scores to each row. Uses data after the cutoff."""
    realised = {b: realised_index(temps_full, sf.year, spec, b) for b in {r["base"] for r in rows}}
    for row in rows:
        fc = sf.forecasts[(row["variant"], row["base"])]
        y = realised[row["base"]]
        row["realised"] = y
        row["error"] = y - fc.mean
        row["z"] = (y - fc.mean) / fc.sd
        row["crps"] = fc.crps(y)
        row["pit"] = fc.cdf(y)
        for level in cfg.evaluation.intervals:
            tag = int(level * 100)
            row[f"hit{tag}"] = bool(row[f"lo{tag}"] <= y <= row[f"hi{tag}"])
        for name, kind, _ in INSTRUMENTS:
            strike = row[f"strike_{name}"]
            payoff = max(y - strike, 0.0) if kind == "call" else max(strike - y, 0.0)
            row[f"payoff_{name}"] = payoff
    return rows


def run_walkforward(
    temps: pd.Series,
    years: list[int],
    variants: list[Variant],
    cfg: Config,
    bases: tuple[float, ...],
    split: str,
) -> pd.DataFrame:
    """Forecast and score every season in ``years``. Each season sees only earlier data."""
    spec = contract_season(cfg)
    rows: list[dict] = []
    for year in years:
        cutoff = spec.start_date(year)
        temps_before = temps.loc[: cutoff - pd.Timedelta(days=1)]
        sf = forecast_season(temps_before, year, variants, cfg, spec, bases)
        season_rows = forecast_rows(sf, cfg)
        rows.extend(score_forecasts(sf, season_rows, temps, spec, cfg))
    frame = pd.DataFrame(rows)
    frame.insert(1, "split", split)
    return frame
