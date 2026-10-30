"""Markdown and JSON output: the results summary, the README block and the holdout log."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

README_START = "<!-- results:start -->"
README_END = "<!-- results:end -->"


def fmt(x: Any, spec: str = ".1f") -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return format(x, spec)


def pm(mean: float, se: float, spec: str = ".1f", signed: bool = False) -> str:
    """'mean ± se' with optional sign."""
    s = ("+" if signed else "") + spec
    return f"{fmt(mean, s)} ± {fmt(se, spec)}"


def md_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    head = "| " + " | ".join(columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    body = ["| " + " | ".join(str(r.get(c, "")) for c in columns) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def short_label(label: str) -> str:
    """'burn|trend=none|window=30' -> 'burn: trend none, window 30'."""
    family, *parts = label.split("|")
    return family + ": " + ", ".join(p.replace("=", " ") for p in parts)


def frame_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """DataFrame to JSON-safe records (NaN -> None, numpy scalars -> Python)."""
    out = []
    for rec in df.to_dict(orient="records"):
        clean = {}
        for k, v in rec.items():
            if isinstance(v, (np.floating, float)):
                v = None if math.isnan(float(v)) else float(v)
            elif isinstance(v, np.integer):
                v = int(v)
            elif isinstance(v, np.bool_):
                v = bool(v)
            clean[k] = v
        out.append(clean)
    return out


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def update_readme(readme: Path, block: str) -> None:
    """Replace the text between the README result markers."""
    text = readme.read_text(encoding="utf-8")
    start = text.index(README_START) + len(README_START)
    end = text.index(README_END)
    readme.write_text(text[:start] + "\n" + block.strip() + "\n" + text[end:], encoding="utf-8")


def append_log(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line.rstrip() + "\n")


def variant_rows(
    table: pd.DataFrame,
    base: float,
    paired: Callable[[str], dict[str, float]] | None,
    order: list[str] | None = None,
) -> list[dict[str, str]]:
    """Format a variant table for markdown; ``paired`` gives CRPS differences vs baseline."""
    sub = table[table["base"] == base]
    if order is not None:
        sub = sub.set_index("variant").loc[order].reset_index()
    rows = []
    for _, r in sub.iterrows():
        row = {
            "Variant": short_label(r["variant"]),
            "n": str(int(r["n"])),
            "CRPS": pm(r["crps"], r["crps_se"]),
            "Bias of E[H]": pm(r["bias"], r["bias_se"], signed=True),
            "RMSE": fmt(r["rmse"]),
            "Mean sd": fmt(r["mean_sd"]),
            "Cover 50/80/90": f"{r['cov50']:.0%} / {r['cov80']:.0%} / {r['cov90']:.0%}",
            "z-RMS": pm(r["z_rms"], r["z_rms_se"], ".2f"),
            "ATM call error (£)": pm(r["err_call_atm"], r["err_call_atm_se"], ".0f", signed=True),
        }
        if paired is not None:
            d = paired(r["variant"])
            row["CRPS vs baseline"] = "-" if d["n"] == 0 or d["se"] == 0 else pm(d["mean"], d["se"], signed=True)
        rows.append(row)
    return rows
