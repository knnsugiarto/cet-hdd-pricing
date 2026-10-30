"""The full study: development walk-forward, frozen selection, holdout, robustness, outputs.

Order matters and is enforced here: the frozen variants are chosen from development
results and written to disk before any holdout season is forecast.
"""

from __future__ import annotations

import datetime as dt
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from hddlab import figures
from hddlab.config import Config
from hddlab.data import ParseReport, slice_hash
from hddlab.daily_model import DailySpec
from hddlab.diagnostics import (
    complete_seasons,
    ljung_box_table,
    seasonal_index_series,
    warm_day_counts,
)
from hddlab.evaluate import (
    FAMILIES,
    Verdict,
    closed_form_check,
    paired_difference,
    pit_counts,
    residual_acf_table,
    select_frozen,
    squared_pricing_error_difference,
    subperiod_best,
    variant_table,
    verdicts,
    worst_seasons,
)
from hddlab.metrics import mean_estimate
from hddlab.report import (
    append_log,
    fmt,
    frame_records,
    md_table,
    pm,
    short_label,
    update_readme,
    variant_rows,
    write_json,
)
from hddlab.walkforward import INSTRUMENTS, build_variants, contract_season, run_walkforward


@dataclass
class StudyResults:
    dev: pd.DataFrame
    holdout: pd.DataFrame
    dev_table: pd.DataFrame
    holdout_table: pd.DataFrame
    frozen: dict[str, str]
    verdicts: list[Verdict]
    info: dict[str, Any]


def git_state(root: Path, ignore: str = "results") -> str:
    """Short commit hash, with '+dirty' if anything outside ``ignore`` is uncommitted.

    Called before the run writes anything. Generated output under ``ignore`` does not
    count: it is rewritten by every run, so it says nothing about the code that ran.
    Returns 'no-git' outside a repository.
    """
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", ".", f":(exclude){ignore}"],
            cwd=root, capture_output=True, text=True, check=True,
        ).stdout.strip()
        return head + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "no-git"


def run_study(
    cfg: Config,
    temps: pd.Series,
    out_dir: Path,
    data_meta: dict[str, Any],
    parse_report: ParseReport | None,
    readme: Path | None,
    make_figures: bool = True,
) -> StudyResults:
    """Run everything and write results to ``out_dir``. Updates ``readme`` if given."""
    started = time.time()
    run_time = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    commit = git_state(cfg.root, ignore=out_dir.name)
    ev, ct = cfg.evaluation, cfg.contract
    tick, lags, levels = ct.tick_gbp, ev.newey_west_lags, ev.intervals
    headline = ct.base_temp
    bases = cfg.all_bases
    spec = contract_season(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"

    # ---- data checks ---------------------------------------------------------------
    seasons = complete_seasons(temps, spec)
    if seasons[-1] < ev.holdout_last:
        raise ValueError(
            f"Last complete season in the data is {seasons[-1]}, but the holdout needs {ev.holdout_last}."
        )
    provisional_from = pd.Timestamp(cfg.data.provisional_from)
    provisional = [y for y in ev.dev_years + ev.holdout_years if spec.end_date(y) >= provisional_from]
    data_hash = slice_hash(temps, cfg.data.evaluation_end)
    warm = warm_day_counts(temps.loc[: cfg.data.evaluation_end], spec, bases)
    h_series = seasonal_index_series(temps, spec, headline)

    # ---- development ---------------------------------------------------------------
    variants = build_variants(cfg)
    print(f"Development: {len(ev.dev_years)} seasons x {len(variants)} variants x {len(bases)} bases")
    dev = run_walkforward(temps, ev.dev_years, variants, cfg, bases, "dev")
    dev_table = variant_table(dev, levels, lags, tick)
    frozen = select_frozen(dev_table, headline)
    write_json(
        out_dir / "frozen.json",
        {
            "rule": "lowest mean development CRPS per family at the headline base temperature (D15)",
            "selected_at": run_time,
            "base_temp": headline,
            "frozen": frozen,
            "baseline": ev.baseline,
        },
    )
    print("Frozen:", frozen)

    # ---- holdout (frozen variants and the baseline only) ------------------------------
    keep = set(frozen.values()) | {ev.baseline}
    hold_variants = [v for v in variants if v.label in keep]
    print(f"Holdout: {len(ev.holdout_years)} seasons x {len(hold_variants)} variants")
    holdout = run_walkforward(temps, ev.holdout_years, hold_variants, cfg, bases, "holdout")
    holdout_table = variant_table(holdout, levels, lags, tick)
    append_log(
        out_dir / "holdout_runs.log",
        f"{run_time} | commit {commit} | data {data_hash[:12]} | frozen "
        + ", ".join(f"{k}={v}" for k, v in frozen.items()),
    )

    # ---- comparisons and robustness --------------------------------------------------
    the_verdicts = verdicts(dev, holdout, frozen, ev.baseline, headline, tick, lags)

    def paired_dev(label: str, base: float = headline) -> dict[str, float]:
        if label == ev.baseline:
            return {"mean": 0.0, "se": 0.0, "t": float("nan"), "n": 0}
        return paired_difference(dev, label, ev.baseline, "crps", base, lags)

    def paired_hold(label: str, base: float = headline) -> dict[str, float]:
        if label == ev.baseline:
            return {"mean": 0.0, "se": 0.0, "t": float("nan"), "n": 0}
        return paired_difference(holdout, label, ev.baseline, "crps", base, lags)

    split_year = (ev.dev_first + ev.dev_last + 1) // 2
    halves = subperiod_best(dev, headline, split_year)
    robust_frozen = select_frozen(dev_table, bases[1]) if len(bases) > 1 else {}
    cf_check = closed_form_check(pd.concat([dev, holdout], ignore_index=True), frozen["daily"])
    acf_table = residual_acf_table(dev, headline)
    fd = next(v for v in variants if v.label == frozen["daily"]).daily
    assert fd is not None
    lb_specs = [DailySpec(fd.window_years, fd.harmonics, p, fd.vol, fd.vol_harmonics) for p in sorted(set(cfg.daily.ar_orders))]
    last_dev_cutoff = spec.start_date(ev.dev_last)
    lb_table = ljung_box_table(temps.loc[: last_dev_cutoff - pd.Timedelta(days=1)], lb_specs)

    loading_rows = []
    for fam in FAMILIES:
        g = holdout[(holdout["variant"] == frozen[fam]) & (holdout["base"] == headline)].sort_values("year")
        for alpha in ct.loadings:
            price = g["price_call_atm"] + alpha * g["payoff_sd_call_atm"]
            err = mean_estimate(((g["payoff_call_atm"] - price) * tick).to_numpy(), lags)
            loading_rows.append(
                {"family": fam, "alpha": alpha, "mean_price_gbp": float(price.mean() * tick),
                 "error_gbp": err.mean, "error_se": err.se}
            )
    loading = pd.DataFrame(loading_rows)

    sq_pairs = {}
    for fa, fb in (("burn", "index"), ("burn", "daily"), ("index", "daily")):
        if frozen[fa] != frozen[fb]:
            sq_pairs[f"{fa} vs {fb}"] = squared_pricing_error_difference(
                holdout, frozen[fa], frozen[fb], "call_atm", headline, tick, lags
            )

    elapsed = time.time() - started
    n_dev_evals = int(dev.groupby(["variant", "base"]).ngroups)
    info: dict[str, Any] = {
        "run_time": run_time,
        "commit": commit,
        "data": {
            **data_meta,
            "dataset_version": cfg.data.dataset_version,
            "used_from": str(temps.index[0].date()),
            "last_date_in_file": str(
                (parse_report.last_date if parse_report is not None else temps.index[-1]).date()
            ),
            "hash_used_slice": data_hash,
            "hash_slice_end": cfg.data.evaluation_end,
            "provisional_seasons": provisional,
            "parse": None if parse_report is None else {
                "rows": parse_report.n_rows,
                "skipped_lines": parse_report.n_skipped_lines,
                "missing_values_in_file": parse_report.n_missing_values,
                "first_date_in_file": str(parse_report.first_date.date()),
            },
        },
        "variants_tried": len(variants),
        "variant_base_evaluations_on_development": n_dev_evals,
        "dev_seasons": [ev.dev_first, ev.dev_last],
        "holdout_seasons": [ev.holdout_first, ev.holdout_last],
        "elapsed_seconds": round(elapsed, 1),
        "n_paths": cfg.daily.n_paths,
    }

    # ---- outputs -------------------------------------------------------------------------
    dev.to_csv(out_dir / "seasons_development.csv", index=False, float_format="%.6g")
    holdout.to_csv(out_dir / "seasons_holdout.csv", index=False, float_format="%.6g")
    write_json(
        out_dir / "metrics.json",
        {
            "info": info,
            "frozen": frozen,
            "frozen_if_selected_at_other_base": robust_frozen,
            "development_best_by_half": halves,
            "verdicts": [v.__dict__ for v in the_verdicts],
            "development": frame_records(dev_table),
            "holdout": frame_records(holdout_table),
            "closed_form_check": frame_records(cf_check),
            "squared_pricing_error_pairs_holdout": sq_pairs,
            "risk_loading_holdout": frame_records(loading),
            "warm_days": warm.reset_index().rename(columns={"index": "month"}).to_dict(orient="records"),
            "worst_holdout_seasons": frame_records(worst_seasons(holdout, {"baseline": ev.baseline, **frozen}, headline, tick)[0]),
        },
    )

    named = {"baseline": ev.baseline, **frozen}
    worst, worst_shares = worst_seasons(holdout, named, headline, tick)
    if make_figures:
        figures.seasonal_index(h_series, ev.dev_years, ev.holdout_years, headline, fig_dir / "seasonal_index.png")
        both = pd.concat([dev, holdout], ignore_index=True)
        figures.walkforward_intervals(both, frozen, ev.holdout_first, headline, fig_dir / "walkforward_intervals.png")
        figures.pit_histograms(dev, named, headline, fig_dir / "pit_development.png")
        figures.closed_form_check(dev, headline, fig_dir / "closed_form_check.png")
        figures.coverage_plot(dev_table, holdout_table, named, levels, headline, fig_dir / "coverage.png")

    summary = _summary_markdown(
        cfg, info, warm, h_series, dev_table, holdout_table, frozen, named, the_verdicts,
        paired_dev, paired_hold, halves, robust_frozen, cf_check, acf_table, lb_table,
        loading, sq_pairs, dev, worst, worst_shares,
    )
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    if readme is not None:
        update_readme(
            readme,
            _readme_block(
                cfg, info, dev_table, holdout_table, named, the_verdicts, paired_dev, paired_hold, cf_check,
                worst, worst_shares,
            ),
        )
    print(f"Done in {time.time() - started:.0f} s. Results in {out_dir}")
    return StudyResults(dev, holdout, dev_table, holdout_table, frozen, the_verdicts, info)


# ---------------------------------------------------------------------------------------
# Markdown builders
# ---------------------------------------------------------------------------------------


def _headline_rows(
    cfg: Config,
    dev_table: pd.DataFrame,
    holdout_table: pd.DataFrame,
    named: dict[str, str],
    paired_dev,
    paired_hold,
) -> list[dict[str, str]]:
    base = cfg.contract.base_temp
    rows = []
    for name, label in named.items():
        d = dev_table[(dev_table["variant"] == label) & (dev_table["base"] == base)].iloc[0]
        h = holdout_table[(holdout_table["variant"] == label) & (holdout_table["base"] == base)].iloc[0]
        pd_, ph = paired_dev(label), paired_hold(label)
        rows.append(
            {
                "Method": (
                    f"**{name}** (same variant as the baseline)"
                    if name != "baseline" and label == named["baseline"]
                    else f"**{name}** ({short_label(label).split(': ', 1)[1]})"
                ),
                "Dev CRPS": fmt(d["crps"]),
                "Dev Δ vs baseline": "-" if pd_["n"] == 0 else pm(pd_["mean"], pd_["se"], signed=True),
                "Holdout CRPS": fmt(h["crps"]),
                "Holdout Δ vs baseline": "-" if ph["n"] == 0 else pm(ph["mean"], ph["se"], signed=True),
                "Holdout bias of E[H]": pm(h["bias"], h["bias_se"], signed=True),
                "90% cover dev / holdout": f"{d['cov90']:.0%} / {h['cov90']:.0%}",
                "z-RMS dev": fmt(d["z_rms"], ".2f"),
                "Holdout ATM call: mean error £ (MDE)": f"{pm(h['err_call_atm'], h['err_call_atm_se'], '.0f', True)} ({fmt(h['err_call_atm_mde'], '.0f')})",
                "Mean ATM price £": fmt(h["price_call_atm"], ".0f"),
            }
        )
    return rows


HEADLINE_COLUMNS = [
    "Method",
    "Dev CRPS",
    "Dev Δ vs baseline",
    "Holdout CRPS",
    "Holdout Δ vs baseline",
    "Holdout bias of E[H]",
    "90% cover dev / holdout",
    "z-RMS dev",
    "Holdout ATM call: mean error £ (MDE)",
    "Mean ATM price £",
]


def _worst_markdown(worst: pd.DataFrame, shares: dict[str, float], named: dict[str, str]) -> str:
    rows = []
    for _, r in worst.iterrows():
        row = {
            "Season": r["season"],
            "Realised H": fmt(r["realised"], ".0f"),
            "vs 30-season reference": f"{r['deviation_sd']:+.1f} sd",
            "ATM call payoff £": fmt(r["atm_call_payoff_gbp"], ".0f"),
        }
        for name in named:
            row[f"{name}: mean / PIT / price £"] = (
                f"{r[f'{name}_mean']:.0f} / {r[f'{name}_pit']:.2f} / {r[f'{name}_atm_price_gbp']:.0f}"
            )
        rows.append(row)
    cols = ["Season", "Realised H", "vs 30-season reference", "ATM call payoff £", *[f"{n}: mean / PIT / price £" for n in named]]
    share_text = ", ".join(f"{n} {v:.0%}" for n, v in shares.items())
    return (
        md_table(rows, cols)
        + "\n\nShare of each method's total squared ATM-call pricing error from the two most extreme "
        + f"holdout seasons: {share_text}."
    )


def _verdict_rows(vs: list[Verdict]) -> list[dict[str, str]]:
    return [{"": v.key, "Prediction": v.prediction, "Evidence": v.evidence, "Verdict": f"**{v.verdict}**"} for v in vs]


def _cf_sentence(cf: pd.DataFrame) -> str:
    parts = []
    for _, r in cf.iterrows():
        parts.append(
            f"base {r['base']:g} °C ({int(r['n'])} seasons): Bachelier-vs-MC z on the linear-index call "
            f"mean {r['z_call_l_mean']:+.2f}, sd {r['z_call_l_sd']:.2f}, max |z| {r['z_call_l_max_abs']:.1f}; "
            f"exact E[H] vs MC z sd {r['z_mean_h_sd']:.2f}, max |z| {r['z_mean_h_max_abs']:.1f}; "
            f"truncation gap {r['truncation_gap_pct_of_price']:.2f}% of the ATM price "
            f"(MC standard error {r['mc_se_pct_of_price']:.1f}%)"
        )
    return "; ".join(parts) + "."


def _readme_block(
    cfg, info, dev_table, holdout_table, named, vs, paired_dev, paired_hold, cf, worst, worst_shares
) -> str:
    d = info["data"]
    ev = cfg.evaluation
    lines = [
        f"*Generated by `scripts/reproduce.py` on {info['run_time']} (commit `{info['commit']}`). "
        f"Data: {d['dataset_version']}, file last date {d['last_date_in_file']}, SHA-256 of the "
        f"{d['used_from']} to {d['hash_slice_end']} slice `{d['hash_used_slice'][:12]}`. "
        f"{info['variants_tried']} variants tried; {ev.dev_last - ev.dev_first + 1} development seasons "
        f"({ev.dev_first}/{str(ev.dev_first + 1)[2:]} to {ev.dev_last}/{str(ev.dev_last + 1)[2:]}), "
        f"{ev.holdout_last - ev.holdout_first + 1} holdout seasons "
        f"({ev.holdout_first}/{str(ev.holdout_first + 1)[2:]} to {ev.holdout_last}/{str(ev.holdout_last + 1)[2:]}). "
        f"Seasons using provisional data: {', '.join(f'{y}/{str(y + 1)[2:]}' for y in d['provisional_seasons']) or 'none'}.*",
        "",
        f"Base {cfg.contract.base_temp:g} °C, tick £{cfg.contract.tick_gbp:g} per HDD, prices undiscounted. "
        "CRPS and bias in HDD; lower CRPS is better; Δ is the paired difference against the baseline "
        "(negative = better), ± one Newey-West standard error. MDE: the smallest bias a 5% test would "
        "detect with 80% power on that sample.",
        "",
        md_table(_headline_rows(cfg, dev_table, holdout_table, named, paired_dev, paired_hold), HEADLINE_COLUMNS),
        "",
        "**Pre-registered predictions** (written 29 Oct 2026, before any real-data run; rules in `docs/decisions.md`):",
        "",
        md_table(_verdict_rows(vs), ["", "Prediction", "Evidence", "Verdict"]),
        "",
        f"**Monte Carlo check:** {_cf_sentence(cf[cf['base'] == cfg.contract.base_temp])}",
        "",
        "**Most extreme holdout seasons** (furthest from the 30-season reference mean):",
        "",
        _worst_markdown(worst, worst_shares, named),
        "",
        "![Forecasts against outcomes](results/figures/walkforward_intervals.png)",
        "",
        "Full tables, all variants, robustness checks and diagnostics: [`results/summary.md`](results/summary.md).",
    ]
    return "\n".join(lines)


def _summary_markdown(
    cfg, info, warm, h_series, dev_table, holdout_table, frozen, named, vs, paired_dev, paired_hold,
    halves, robust_frozen, cf, acf_table, lb_table, loading, sq_pairs, dev, worst, worst_shares,
) -> str:
    ev, ct = cfg.evaluation, cfg.contract
    base = ct.base_temp
    d = info["data"]
    dev_h = h_series.loc[ev.dev_first : ev.dev_last]
    out = [
        "# Results summary",
        "",
        f"Generated by `scripts/reproduce.py` on {info['run_time']}, commit `{info['commit']}`, "
        f"in {info['elapsed_seconds']} s with {info['n_paths']} Monte Carlo paths per season and variant.",
        "",
        "## Data",
        "",
        f"- Dataset: {d['dataset_version']}; used from {d['used_from']}; last date in file {d['last_date_in_file']}.",
        f"- SHA-256 of the used slice ({d['used_from']} to {d['hash_slice_end']}): `{d['hash_used_slice']}`.",
        f"- Download: {d.get('downloaded_utc', 'n/a')} from {d.get('url', 'n/a')}.",
        f"- Seasons containing provisional values: {d['provisional_seasons'] or 'none'}.",
        f"- Development-period seasonal index (base {base:g} °C): mean {dev_h.mean():.0f} HDD, "
        f"sd {dev_h.std(ddof=1):.0f} HDD over {len(dev_h)} seasons.",
        "",
        "Days at or above each base temperature in complete Nov-Mar seasons used (1900/01 onwards):",
        "",
        md_table([{"Month": m, **{c: str(v) for c, v in r.items()}} for m, r in warm.iterrows()], ["Month", *warm.columns]),
        "",
        "## Variants tried",
        "",
        f"{info['variants_tried']} variants (burn and index: trend none/linear x window 30/50; daily: AR order "
        f"{'/'.join(map(str, cfg.daily.ar_orders))} x harmonics {'/'.join(map(str, cfg.daily.harmonics))} x "
        f"volatility {'/'.join(cfg.daily.vol)} x window {'/'.join(map(str, cfg.daily.windows))}), each evaluated at "
        f"base temperatures {', '.join(f'{b:g}' for b in cfg.all_bases)} °C: "
        f"{info['variant_base_evaluations_on_development']} development evaluations. "
        "Selection used only the headline base.",
        "",
        f"## Development seasons, base {base:g} °C, all variants",
        "",
        md_table(
            variant_rows(dev_table.sort_values(["family", "crps"]), base, paired_dev),
            ["Variant", "n", "CRPS", "CRPS vs baseline", "Bias of E[H]", "RMSE", "Mean sd", "Cover 50/80/90", "z-RMS", "ATM call error (£)"],
        ),
        "",
        "Frozen by the pre-registered rule (lowest development CRPS per family):",
        "",
        *[f"- {fam}: {short_label(lab)}" for fam, lab in frozen.items()],
        "",
        f"## Holdout seasons, base {base:g} °C",
        "",
        md_table(
            variant_rows(holdout_table, base, paired_hold, order=list(dict.fromkeys(named.values()))),
            ["Variant", "n", "CRPS", "CRPS vs baseline", "Bias of E[H]", "RMSE", "Mean sd", "Cover 50/80/90", "z-RMS", "ATM call error (£)"],
        ),
        "",
        "### Pricing errors by instrument (holdout, £ per contract, payoff minus price)",
        "",
    ]
    rows = []
    for name, label in named.items():
        r = holdout_table[(holdout_table["variant"] == label) & (holdout_table["base"] == base)].iloc[0]
        row = {"Method": name}
        for inst, _, _ in INSTRUMENTS:
            row[inst] = f"{pm(r[f'err_{inst}'], r[f'err_{inst}_se'], '.0f', True)} (price {r[f'price_{inst}']:.0f}, MDE {r[f'err_{inst}_mde']:.0f})"
        rows.append(row)
    out += [md_table(rows, ["Method", *[i[0] for i in INSTRUMENTS]]), ""]
    out += [
        "lo/atm/hi: strikes at the reference mean -1, 0, +1 reference sd (previous 30 raw seasons).",
        "",
        "### Most extreme holdout seasons",
        "",
        _worst_markdown(worst, worst_shares, named),
        "",
        "### Which method prices the ATM call better? (holdout, paired squared errors)",
        "",
        md_table(
            [{"Pair": k, "Mean difference (£²)": fmt(v["mean"], ".0f"), "SE": fmt(v["se"], ".0f"), "t": fmt(v["t"], "+.2f"), "n": str(v["n"])} for k, v in sq_pairs.items()],
            ["Pair", "Mean difference (£²)", "SE", "t", "n"],
        ),
        "",
        "## Pre-registered predictions",
        "",
        md_table(
            [{"": v.key, "Prediction": v.prediction, "Rule": v.rule, "Evidence": v.evidence, "Verdict": f"**{v.verdict}**"} for v in vs],
            ["", "Prediction", "Rule", "Evidence", "Verdict"],
        ),
        "",
        "## Robustness",
        "",
        f"**Base temperature.** Frozen variants re-evaluated at {', '.join(f'{b:g}' for b in cfg.robustness_bases)} °C "
        "(selection not repeated):",
        "",
    ]
    for rb in cfg.robustness_bases:
        out += [
            md_table(
                variant_rows(holdout_table, rb, lambda lab, b=rb: paired_hold(lab, b), order=list(dict.fromkeys(named.values()))),
                ["Variant", "n", "CRPS", "CRPS vs baseline", "Bias of E[H]", "Cover 50/80/90", "z-RMS", "ATM call error (£)"],
            ),
            "",
        ]
    if robust_frozen:
        same = all(robust_frozen[f] == frozen[f] for f in FAMILIES)
        out += [
            f"Had selection used base {cfg.robustness_bases[0]:g} °C instead, it would have picked: "
            + "; ".join(short_label(robust_frozen[f]) for f in FAMILIES)
            + (" (the same variants)." if same else " (different from the headline choice)."),
            "",
        ]
    out += [
        "**Sub-periods.** Best variant per family by development CRPS in each half:",
        "",
        md_table(
            [{"Period": p, **{f: short_label(lab) for f, lab in best.items()}} for p, best in halves.items()],
            ["Period", *FAMILIES],
        ),
        "",
        "**Window and trend form** are part of the variant grid above.",
        "",
        "**Risk loading** (holdout, ATM call; price = E[payoff] + alpha sd[payoff]):",
        "",
        md_table(
            [{"Method": r["family"], "alpha": fmt(r["alpha"], ".2f"), "Mean price £": fmt(r["mean_price_gbp"], ".0f"), "Mean error £": pm(r["error_gbp"], r["error_se"], ".0f", True)} for _, r in loading.iterrows()],
            ["Method", "alpha", "Mean price £", "Mean error £"],
        ),
        "",
        "## Monte Carlo against closed forms (frozen daily variant, development and holdout seasons)",
        "",
        _cf_sentence(cf),
        "",
        "## Daily model diagnostics",
        "",
        "Mean over development seasons of the AR coefficient phi_1 and residual autocorrelations:",
        "",
        md_table(
            [{"Variant": short_label(r["variant"]), **{c: fmt(r[c], "+.3f") for c in acf_table.columns if c != "variant"}} for _, r in acf_table.iterrows()],
            ["Variant", *[c for c in acf_table.columns if c != "variant"]],
        ),
        "",
        f"Ljung-Box (lag 10) on the fit used for the last development season ({ev.dev_last}/{str(ev.dev_last + 1)[2:]}). "
        "With about 10,000 residuals it rejects even tiny autocorrelation; read it alongside the autocorrelations.",
        "",
        md_table(
            [
                {c: (short_label(v) if c == "variant" else fmt(v, ".3g") if isinstance(v, float) else str(v)) for c, v in r.items()}
                for r in lb_table.to_dict(orient="records")
            ],
            list(lb_table.columns),
        ),
        "",
        "## PIT histograms (development, 10 bins)",
        "",
        md_table(
            [{"Method": name, "Counts": " ".join(str(c) for c in pit_counts(dev, label, base))} for name, label in named.items()],
            ["Method", "Counts"],
        ),
        "",
        "Figures: `results/figures/`.",
        "",
    ]
    return "\n".join(out)
