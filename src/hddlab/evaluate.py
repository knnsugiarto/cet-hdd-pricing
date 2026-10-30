"""Aggregate season-level results: scores, calibration, pricing errors, selection, verdicts.

Every function takes the per-season frames from ``run_walkforward`` and returns plain
tables. Pricing errors are reported in pounds: (payoff - price) * tick, undiscounted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from hddlab.metrics import (
    coverage,
    mean_estimate,
    pit_histogram,
    standardised_rms,
)
from hddlab.walkforward import INSTRUMENTS

FAMILIES = ("burn", "index", "daily")


def pricing_errors(frame: pd.DataFrame, instrument: str, tick: float) -> np.ndarray:
    """Realised payoff minus price, in pounds, one value per row."""
    return (frame[f"payoff_{instrument}"] - frame[f"price_{instrument}"]).to_numpy() * tick


def variant_table(
    frame: pd.DataFrame, levels: tuple[float, ...], lags: int, tick: float
) -> pd.DataFrame:
    """One row per (base, variant): CRPS, bias, RMSE, coverage, z-RMS and option pricing errors."""
    records = []
    for (base, variant), g in frame.groupby(["base", "variant"], sort=False):
        g = g.sort_values("year")
        crps = mean_estimate(g["crps"].to_numpy(), lags)
        bias = mean_estimate(g["error"].to_numpy(), lags)
        z_rms, z_rms_se = standardised_rms(g["z"].to_numpy())
        rec = {
            "base": base,
            "family": g["family"].iloc[0],
            "variant": variant,
            "n": len(g),
            "crps": crps.mean,
            "crps_se": crps.se,
            "bias": bias.mean,
            "bias_se": bias.se,
            "rmse": float(np.sqrt(np.mean(g["error"] ** 2))),
            "mean_sd": float(g["sd"].mean()),
            "z_rms": z_rms,
            "z_rms_se": z_rms_se,
        }
        for level in levels:
            tag = int(level * 100)
            rec[f"cov{tag}"] = coverage(g[f"hit{tag}"].to_numpy(), level).rate
        for name, _, _ in INSTRUMENTS:
            err = mean_estimate(pricing_errors(g, name, tick), lags)
            rec[f"err_{name}"] = err.mean
            rec[f"err_{name}_se"] = err.se
            rec[f"err_{name}_mde"] = err.mde
            rec[f"price_{name}"] = float(g[f"price_{name}"].mean() * tick)
        records.append(rec)
    return pd.DataFrame(records)


def paired_difference(
    frame: pd.DataFrame, a: str, b: str, column: str, base: float, lags: int
) -> dict[str, float]:
    """Mean over seasons of column(a) - column(b), with Newey-West SE. Seasons are matched."""
    sub = frame[frame["base"] == base]
    left = sub[sub["variant"] == a].set_index("year")[column]
    right = sub[sub["variant"] == b].set_index("year")[column]
    joined = pd.concat([left, right], axis=1, join="inner", keys=["a", "b"]).sort_index()
    est = mean_estimate((joined["a"] - joined["b"]).to_numpy(), lags)
    return {"mean": est.mean, "se": est.se, "t": est.t, "n": est.n}


def squared_pricing_error_difference(
    frame: pd.DataFrame, a: str, b: str, instrument: str, base: float, tick: float, lags: int
) -> dict[str, float]:
    """Paired difference of squared pricing errors (pounds^2): which method prices better.

    Squared error is a consistent score for an expected value, so a negative mean means
    method a's prices were closer to realised payoffs than method b's.
    """
    sub = frame[frame["base"] == base].copy()
    sub["sq"] = pricing_errors(sub, instrument, tick) ** 2
    return paired_difference(sub, a, b, "sq", base, lags)


def select_frozen(dev_table: pd.DataFrame, base: float) -> dict[str, str]:
    """Pre-registered rule (decision D15): in each family, the variant with the lowest
    mean development CRPS at the headline base temperature."""
    sub = dev_table[dev_table["base"] == base]
    return {
        fam: str(sub[sub["family"] == fam].sort_values("crps").iloc[0]["variant"])
        for fam in FAMILIES
    }


def pit_counts(frame: pd.DataFrame, variant: str, base: float, bins: int = 10) -> list[int]:
    sub = frame[(frame["variant"] == variant) & (frame["base"] == base)]
    return pit_histogram(sub["pit"].to_numpy(), bins).tolist()


def subperiod_best(
    dev: pd.DataFrame, base: float, split_year: int
) -> dict[str, dict[str, str]]:
    """Best variant per family by CRPS in each half of the development period."""
    out: dict[str, dict[str, str]] = {}
    sub = dev[dev["base"] == base]
    for name, part in (
        (f"{sub['year'].min()}-{split_year - 1}", sub[sub["year"] < split_year]),
        (f"{split_year}-{sub['year'].max()}", sub[sub["year"] >= split_year]),
    ):
        means = part.groupby(["family", "variant"])["crps"].mean().reset_index()
        out[name] = {
            fam: str(means[means["family"] == fam].sort_values("crps").iloc[0]["variant"])
            for fam in FAMILIES
        }
    return out


def closed_form_check(frame: pd.DataFrame, daily_label: str) -> pd.DataFrame:
    """Monte Carlo vs exact Gaussian results for one daily variant, per base temperature.

    Each season uses its own independent draws, so across seasons the z-scores should
    look like independent N(0, 1) values. (Variants within a season share draws, so
    pooling variants would not add independent evidence.) The truncation gap is the
    call on H minus the call on L on the same paths: the error of treating H as Gaussian.
    """
    daily = frame[frame["variant"] == daily_label]
    records = []
    for base, g in daily.groupby("base"):
        rec: dict[str, float] = {"base": base, "n": len(g)}
        for col in ("z_mean_l", "z_call_l", "z_mean_h"):
            z = g[col].to_numpy()
            rec[f"{col}_mean"] = float(z.mean())
            rec[f"{col}_sd"] = float(z.std(ddof=1))
            rec[f"{col}_max_abs"] = float(np.abs(z).max())
        rec["truncation_gap_pct_of_price"] = float(
            100.0 * (g["truncation_gap_atm"] / g["price_call_atm"]).mean()
        )
        rec["mc_se_pct_of_price"] = float(
            100.0 * (g["mc_se_call_atm"] / g["price_call_atm"]).mean()
        )
        records.append(rec)
    return pd.DataFrame(records)


def residual_acf_table(frame: pd.DataFrame, base: float) -> pd.DataFrame:
    """Mean residual autocorrelations of each daily variant across seasons."""
    daily = frame[(frame["family"] == "daily") & (frame["base"] == base)]
    cols = [c for c in daily.columns if c.startswith("resid_acf")]
    return daily.groupby("variant")[["phi1", *cols]].mean().reset_index()


@dataclass(frozen=True)
class Verdict:
    key: str
    prediction: str
    rule: str
    evidence: str
    verdict: str


def verdicts(
    dev: pd.DataFrame,
    holdout: pd.DataFrame,
    frozen: dict[str, str],
    baseline: str,
    base: float,
    tick: float,
    lags: int,
) -> list[Verdict]:
    """Apply the pre-registered decision rules for P1-P3 (docs/decisions.md, D16)."""
    out = []

    # P1: undetrended burn overestimates E[H] in recent decades.
    hb = holdout[(holdout["variant"] == baseline) & (holdout["base"] == base)].sort_values("year")
    b1 = mean_estimate(hb["error"].to_numpy(), lags)
    late_dev = dev[(dev["variant"] == baseline) & (dev["base"] == base) & (dev["year"] >= 1980)]
    late_note = "no development seasons from 1980"
    if len(late_dev) > 1:
        b1_dev = mean_estimate(late_dev.sort_values("year")["error"].to_numpy(), lags)
        late_note = f"development seasons 1980-{late_dev['year'].max()}: {b1_dev.mean:+.1f} HDD (SE {b1_dev.se:.1f})"
    if b1.t < -1.645:
        v1 = "supported"
    elif b1.mean < 0:
        v1 = "right direction, not significant"
    else:
        v1 = "not supported"
    out.append(
        Verdict(
            "P1",
            "Undetrended burn overestimates E[H] in recent decades.",
            "Holdout mean of (realised H - forecast mean) for the baseline; supported if t < -1.645.",
            f"holdout {b1.mean:+.1f} HDD (SE {b1.se:.1f}, t {b1.t:+.2f}, n {b1.n}); {late_note}",
            v1,
        )
    )

    # P2: the daily model's intervals are too narrow.
    dd = dev[(dev["variant"] == frozen["daily"]) & (dev["base"] == base)]
    cov90 = coverage(dd["hit90"].to_numpy(), 0.9)
    z_rms, z_rms_se = standardised_rms(dd["z"].to_numpy())
    hd = holdout[(holdout["variant"] == frozen["daily"]) & (holdout["base"] == base)]
    cov90_h = coverage(hd["hit90"].to_numpy(), 0.9)
    v2 = "supported" if cov90.z < -1.645 else "not supported"
    out.append(
        Verdict(
            "P2",
            "The daily AR model's predictive sd of H is too small, so its 90% intervals under-cover.",
            "Frozen daily variant, development seasons: supported if 90% coverage is below 0.9 "
            "by more than 1.645 binomial SEs.",
            f"development 90% coverage {cov90.rate:.0%} of {cov90.n} (z {cov90.z:+.2f}); "
            f"z-RMS {z_rms:.2f} (SE {z_rms_se:.2f}); holdout 90% coverage {cov90_h.rate:.0%} of {cov90_h.n}",
            v2,
        )
    )

    # P3: on the holdout, no method prices the ATM call significantly better than another.
    pairs = [("burn", "index"), ("burn", "daily"), ("index", "daily")]
    parts = []
    any_sig = False
    for fa, fb in pairs:
        if frozen[fa] == frozen[fb]:
            continue
        d = squared_pricing_error_difference(
            holdout, frozen[fa], frozen[fb], "call_atm", base, tick, lags
        )
        any_sig = any_sig or abs(d["t"]) >= 1.96
        parts.append(f"{fa} vs {fb}: t {d['t']:+.2f}")
    out.append(
        Verdict(
            "P3",
            "On the holdout, no pair of frozen methods differs significantly in squared "
            "ATM-call pricing error.",
            "Paired Newey-West t-test on squared pricing errors; supported if every |t| < 1.96.",
            "; ".join(parts),
            "not supported" if any_sig else "supported",
        )
    )
    return out


def worst_seasons(
    holdout: pd.DataFrame, named: dict[str, str], base: float, tick: float, n: int = 5
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Holdout seasons furthest from the 30-season reference mean, with each method's view.

    Returns the table and, per method, the share of its total squared ATM-call pricing
    error that comes from the two most extreme seasons (how much a few winters dominate).
    """
    sub = holdout[holdout["base"] == base]
    ref = sub[sub["variant"] == named["baseline"]].set_index("year")
    ref_sd = ref["strike_call_hi"] - ref["strike_call_atm"]
    deviation = (ref["realised"] - ref["strike_call_atm"]) / ref_sd
    order = deviation.abs().sort_values(ascending=False).index[:n]
    rows = []
    for year in order:
        row = {
            "season": f"{year}/{str(year + 1)[2:]}",
            "realised": float(ref.loc[year, "realised"]),
            "reference_mean": float(ref.loc[year, "strike_call_atm"]),
            "deviation_sd": float(deviation.loc[year]),
            "atm_call_payoff_gbp": float(ref.loc[year, "payoff_call_atm"] * tick),
        }
        for name, label in named.items():
            r = sub[(sub["variant"] == label) & (sub["year"] == year)].iloc[0]
            row[f"{name}_mean"] = float(r["mean"])
            row[f"{name}_pit"] = float(r["pit"])
            row[f"{name}_atm_price_gbp"] = float(r["price_call_atm"] * tick)
        rows.append(row)
    shares = {}
    top2 = list(order[:2])
    for name, label in named.items():
        g = sub[sub["variant"] == label].set_index("year")
        sq = ((g["payoff_call_atm"] - g["price_call_atm"]) * tick) ** 2
        shares[name] = float(sq.loc[top2].sum() / sq.sum())
    return pd.DataFrame(rows), shares
