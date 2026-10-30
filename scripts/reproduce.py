"""Reproduce every result in the README from the downloaded data.

Usage:
    python scripts/reproduce.py               # real data; writes results/ and updates README.md
    python scripts/reproduce.py --synthetic   # smoke run on synthetic data; writes results-synthetic/ only

Takes a few minutes on a laptop CPU.
"""

from __future__ import annotations

import argparse
import json
import sys

from hddlab.config import DEFAULT_CONFIG, load_config
from hddlab.data import load_daily, restrict_and_validate
from hddlab.pipeline import run_study
from hddlab.synthetic import synthetic_cet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data; never touches README or results/.")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.synthetic:
        raw, parse_report = synthetic_cet(), None
        meta = {"source": "synthetic (hddlab.synthetic); not real data"}
        out_dir, readme = cfg.root / "results-synthetic", None
    else:
        raw, parse_report = load_daily(cfg.data.raw_path)
        meta = json.loads(cfg.data.meta_path.read_text()) if cfg.data.meta_path.exists() else {}
        out_dir, readme = cfg.root / "results", cfg.root / "README.md"

    # The study needs nothing after evaluation_end, so later (provisional) days cannot affect it.
    temps = restrict_and_validate(raw.loc[: cfg.data.evaluation_end], cfg.data.start_date)
    run_study(cfg, temps, out_dir, meta, parse_report, readme, make_figures=not args.no_figures)
    return 0


if __name__ == "__main__":
    sys.exit(main())
