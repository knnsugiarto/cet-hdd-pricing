"""Pre-registered forecast of a season that has not started, and its later scoring.

Make a forecast (commit the output to git before the season starts):
    python scripts/forecast.py make --season-start 2026-12-01 --season-end 2027-03-31

Score it once the season's data are published:
    python scripts/forecast.py score forecasts/2026-27_1201-0331.json

The forecast uses the frozen variants in results/frozen.json and only the data in the
downloaded file, which ends a couple of days before the download date.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from hddlab.config import DEFAULT_CONFIG, Config, load_config
from hddlab.data import load_daily, restrict_and_validate, slice_hash
from hddlab.distributions import Forecast
from hddlab import figures
from hddlab.seasons import SeasonSpec, realised_index
from hddlab.walkforward import INSTRUMENTS, build_variants, forecast_rows, forecast_season


def _load_temps(cfg: Config) -> pd.Series:
    raw, _ = load_daily(cfg.data.raw_path)
    last_valid = raw.last_valid_index()
    return restrict_and_validate(raw.loc[:last_valid], cfg.data.start_date)


def _stem(spec: SeasonSpec, year: int) -> str:
    return f"{year}-{str(year + 1)[2:]}_{spec.start[0]:02d}{spec.start[1]:02d}-{spec.end[0]:02d}{spec.end[1]:02d}"


def make(cfg: Config, start: str, end: str, allow_late: bool) -> Path:
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    spec = SeasonSpec((start_ts.month, start_ts.day), (end_ts.month, end_ts.day))
    year = start_ts.year
    if spec.end_date(year) != end_ts:
        raise SystemExit("Season end must fall in the year after the season start.")
    temps = _load_temps(cfg)
    if temps.index[-1] >= start_ts and not allow_late:
        raise SystemExit(
            f"The data already reach {temps.index[-1].date()}, inside the season. "
            "A pre-registered forecast must be made before the season starts."
        )
    temps = temps.loc[: start_ts - pd.Timedelta(days=1)]
    frozen_path = cfg.root / "results" / "frozen.json"
    frozen_doc = json.loads(frozen_path.read_text())
    keep = set(frozen_doc["frozen"].values()) | {frozen_doc["baseline"]}
    variants = [v for v in build_variants(cfg) if v.label in keep]
    base = cfg.contract.base_temp
    tick = cfg.contract.tick_gbp

    sf = forecast_season(temps, year, variants, cfg, spec, (base,))
    rows = {r["variant"]: r for r in forecast_rows(sf, cfg)}
    created = dt.datetime.now(dt.timezone.utc)
    years_to_payment = (end_ts - pd.Timestamp(created.date())).days / 365.0
    discount = float(np.exp(-cfg.contract.discount_rate * years_to_payment))

    names = {"baseline": frozen_doc["baseline"], **frozen_doc["frozen"]}
    methods = {}
    loading_rows = []
    for name, label in names.items():
        fc: Forecast = sf.forecasts[(label, base)]
        row = rows[label]
        entry = {
            "variant": label,
            "kind": fc.kind,
            "mean_hdd": fc.mean,
            "sd_hdd": fc.sd,
            "quantiles_hdd": {str(q): fc.quantile(q) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
            "instruments": {},
        }
        if fc.samples is not None:
            entry["samples_hdd"] = [round(float(x), 1) for x in fc.samples]
        for inst, kind, mult in INSTRUMENTS:
            price, payoff_sd = row[f"price_{inst}"], row[f"payoff_sd_{inst}"]
            entry["instruments"][inst] = {
                "type": kind,
                "strike_hdd": row[f"strike_{inst}"],
                "price_gbp_undiscounted": price * tick,
                "price_gbp_pv": price * tick * discount,
                "payoff_sd_gbp": payoff_sd * tick,
                "loaded_price_gbp_undiscounted": {
                    str(a): (price + a * payoff_sd) * tick for a in cfg.contract.loadings
                },
            }
        if name != "baseline" or label not in frozen_doc["frozen"].values():
            for a in cfg.contract.loadings:
                loading_rows.append(
                    {"family": name, "alpha": a, "price_gbp": (row["price_call_atm"] + a * row["payoff_sd_call_atm"]) * tick}
                )
        methods[name] = entry

    doc = {
        "created_utc": created.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "season": {"start": start, "end": end, "days": spec.length(year)},
        "contract": {"base_temp_c": base, "tick_gbp": tick, "payment": end},
        "data": {
            "last_date_used": str(temps.index[-1].date()),
            "sha256_used_slice": slice_hash(temps, str(temps.index[-1].date())),
            "dataset_version": cfg.data.dataset_version,
        },
        "discount": {"rate_assumption": cfg.contract.discount_rate, "factor": discount},
        "strikes_rule": f"mean +/- sd of the previous {cfg.contract.strike_reference_seasons} raw seasonal totals",
        "methods": methods,
    }
    cfg.forecast_dir.mkdir(parents=True, exist_ok=True)
    stem = _stem(spec, year)
    json_path = cfg.forecast_dir / f"{stem}.json"
    json_path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    figures.risk_loading(pd.DataFrame(loading_rows), cfg.forecast_dir / f"{stem}_risk_loading.png")
    (cfg.forecast_dir / f"{stem}.md").write_text(_markdown(doc), encoding="utf-8")
    print(_markdown(doc))
    print(f"Wrote {json_path.relative_to(cfg.root)}. Commit it before {start}.")
    return json_path


def _markdown(doc: dict) -> str:
    s = doc["season"]
    lines = [
        f"# Pre-registered forecast: HDD index, {s['start']} to {s['end']}",
        "",
        f"Created {doc['created_utc']} from data up to {doc['data']['last_date_used']} "
        f"(slice hash `{doc['data']['sha256_used_slice'][:12]}`). Base {doc['contract']['base_temp_c']:g} °C, "
        f"tick £{doc['contract']['tick_gbp']:g}. Undiscounted prices; the present value uses an assumed "
        f"{doc['discount']['rate_assumption']:.1%} rate (factor {doc['discount']['factor']:.4f}).",
        "",
        "| Method | Variant | E[H] | sd | 5% | 50% | 95% | ATM strike | ATM call £ | ATM put £ |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, m in doc["methods"].items():
        q = m["quantiles_hdd"]
        c, p = m["instruments"]["call_atm"], m["instruments"]["put_atm"]
        lines.append(
            f"| {name} | {m['variant'].replace('|', ', ')} | {m['mean_hdd']:.0f} | {m['sd_hdd']:.0f} | "
            f"{q['0.05']:.0f} | {q['0.5']:.0f} | {q['0.95']:.0f} | {c['strike_hdd']:.0f} | "
            f"{c['price_gbp_undiscounted']:.0f} | {p['price_gbp_undiscounted']:.0f} |"
        )
    lines += [
        "",
        "One season is a single draw: whatever happens, it shows the forecast was made in advance, "
        "not that a method is accurate.",
        "",
    ]
    return "\n".join(lines)


def score(cfg: Config, path: Path) -> None:
    doc = json.loads(path.read_text())
    start, end = pd.Timestamp(doc["season"]["start"]), pd.Timestamp(doc["season"]["end"])
    spec = SeasonSpec((start.month, start.day), (end.month, end.day))
    temps = _load_temps(cfg)
    if temps.index[-1] < end:
        raise SystemExit(f"Data end {temps.index[-1].date()}; the season ends {end.date()}.")
    base = doc["contract"]["base_temp_c"]
    tick = doc["contract"]["tick_gbp"]
    y = realised_index(temps, start.year, spec, base)
    result = {"scored_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "realised_hdd": y, "methods": {}}
    for name, m in doc["methods"].items():
        if "samples_hdd" in m:
            fc = Forecast.ensemble(np.array(m["samples_hdd"]))
        else:
            fc = Forecast.gaussian(m["mean_hdd"], m["sd_hdd"])
        errors = {}
        for inst, info in m["instruments"].items():
            k = info["strike_hdd"]
            payoff = max(y - k, 0.0) if info["type"] == "call" else max(k - y, 0.0)
            errors[inst] = payoff * tick - info["price_gbp_undiscounted"]
        result["methods"][name] = {"error_hdd": y - m["mean_hdd"], "crps": fc.crps(y), "pit": fc.cdf(y), "pricing_error_gbp": errors}
    out = path.with_name(path.stem + "_score.json")
    out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=1))
    print(f"Wrote {out.relative_to(cfg.root)}.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    sub = parser.add_subparsers(dest="command", required=True)
    mk = sub.add_parser("make")
    mk.add_argument("--season-start", required=True)
    mk.add_argument("--season-end", required=True)
    mk.add_argument("--allow-late", action="store_true", help="Forecast even if the season has started (not pre-registered).")
    sc = sub.add_parser("score")
    sc.add_argument("forecast_json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.command == "make":
        make(cfg, args.season_start, args.season_end, args.allow_late)
    else:
        score(cfg, Path(args.forecast_json))
    return 0


if __name__ == "__main__":
    sys.exit(main())
