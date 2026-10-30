"""Download the HadCET daily mean temperature series and record its provenance.

Usage:
    python scripts/download_data.py                 # fetch from the Met Office
    python scripts/download_data.py --file PATH     # use a copy you saved from the browser

Writes data/raw/meantemp_daily_totals.txt (never committed) and data/raw/download_meta.json.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

from hddlab.config import DEFAULT_CONFIG, load_config
from hddlab.data import parse_daily_text, restrict_and_validate


def fetch(url: str) -> bytes:
    """GET ``url`` with a certificate bundle that works on python.org macOS builds."""
    try:
        import certifi

        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()
    request = urllib.request.Request(url, headers={"User-Agent": "hddlab-research/0.1"})
    with urllib.request.urlopen(request, context=context, timeout=60) as response:
        return response.read()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--file", help="Use this local copy of meantemp_daily_totals.txt instead of downloading.")
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.file:
        payload = Path(args.file).read_bytes()
        source = f"local file {Path(args.file).name} (saved from {cfg.data.url})"
    else:
        print(f"Downloading {cfg.data.url}")
        try:
            payload = fetch(cfg.data.url)
        except (urllib.error.URLError, ssl.SSLError) as exc:
            print(f"Download failed: {exc}", file=sys.stderr)
            print(
                "Fix: open the URL in a browser, save the page as a .txt file, then run\n"
                "    python scripts/download_data.py --file /path/to/meantemp_daily_totals.txt",
                file=sys.stderr,
            )
            return 1
        source = cfg.data.url

    text = payload.decode("utf-8", errors="replace")
    series, report = parse_daily_text(text)
    used = restrict_and_validate(series.loc[: cfg.data.evaluation_end], cfg.data.start_date)

    cfg.data.raw_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.data.raw_path.write_bytes(payload)
    meta = {
        "url": cfg.data.url,
        "source": source,
        "downloaded_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "dataset_version_note": cfg.data.dataset_version,
        "file_sha256": hashlib.sha256(payload).hexdigest(),
        "file_bytes": len(payload),
        "rows": report.n_rows,
        "skipped_lines": report.n_skipped_lines,
        "missing_values": report.n_missing_values,
        "first_date": str(report.first_date.date()),
        "last_date": str(report.last_date.date()),
    }
    cfg.data.meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"Saved {cfg.data.raw_path.relative_to(cfg.root)} ({len(payload):,} bytes)")
    print(f"Rows {report.n_rows:,} from {report.first_date.date()} to {report.last_date.date()}")
    print(f"Skipped non-data lines: {report.n_skipped_lines}; missing values coded -99.9: {report.n_missing_values}")
    print(f"Checked {len(used):,} days from {cfg.data.start_date} to {cfg.data.evaluation_end}: no gaps, no missing values")
    return 0


if __name__ == "__main__":
    sys.exit(main())
