"""Check that the test suite catches known, meaningful bugs.

Each mutation edits one line in a temporary copy of the package and runs the tests
against it. A mutation is "caught" if at least one test fails.

Usage:
    python scripts/mutation_check.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (description, file under src/hddlab, original text, mutated text)
MUTATIONS: list[tuple[str, str, str, str]] = [
    ("HDD sign flipped", "seasons.py", "np.maximum(base - np.asarray(temps, dtype=float), 0.0)", "np.maximum(np.asarray(temps, dtype=float) - base, 0.0)"),
    ("leap year taken from the wrong calendar year", "seasons.py", "pd.Timestamp(year=year + 1, month=1, day=1).is_leap_year", "pd.Timestamp(year=year, month=1, day=1).is_leap_year"),
    ("29 Feb proxy never added", "seasons.py", "        if target_has_feb29 and not has_feb29:", "        if False:"),
    ("Bachelier call loses its time-value term", "distributions.py", "return float((mean - strike) * ndtr(d) + sd * _pdf(d))", "return float((mean - strike) * ndtr(d))"),
    ("ensemble CRPS spread term sign flipped", "distributions.py", "return float(term1 - half_mean_abs_diff)", "return float(term1 + half_mean_abs_diff)"),
    ("index sd ignores estimation error", "seasonal_methods.py", "sd = s * np.sqrt(1.0 + 1.0 / n)\n", "sd = s\n"),
    ("detrending projects the wrong way", "seasonal_methods.py", "return totals + slope * (target_year - np.asarray(years, dtype=float))", "return totals - slope * (target_year - np.asarray(years, dtype=float))"),
    ("simulator never updates the AR state", "daily_model.py", "        state[:, 0] = x_new", "        pass"),
    ("variance recursion sums noise on non-season days", "daily_model.py", "        g[p] = m * sig[j]", "        g[p] = sig[j]"),
    ("AR fit uses lags shifted by one day", "daily_model.py", "lags = np.column_stack([x[order - i : n - i] for i in range(1, order + 1)])", "lags = np.column_stack([x[order - i + 1 : n - i + 1] for i in range(1, order + 1)])"),
    ("burn window includes the season being priced", "walkforward.py", "    years = list(range(year - n, year))", "    years = list(range(year - n + 1, year + 1))"),
    ("pricing cutoff includes the first season day", "walkforward.py", "        temps_before = temps.loc[: cutoff - pd.Timedelta(days=1)]", "        temps_before = temps.loc[:cutoff]"),
    ("Newey-West ignores autocorrelation", "metrics.py", "        long_run_var += 2.0 * weight * np.dot(d[:-lag], d[lag:]) / n", "        long_run_var += 0.0"),
    ("missing-value code not converted", "data.py", "    series[missing] = np.nan", "    pass"),
]


def run_tests(src: Path) -> bool:
    """Return True if the test suite passes against the package in ``src``."""
    env = {**os.environ, "PYTHONPATH": str(src)}
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", str(ROOT / "tests")],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )
    return result.returncode == 0


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "src"
        shutil.copytree(ROOT / "src", src)
        # The package finds its config and scripts relative to itself.
        shutil.copytree(ROOT / "configs", Path(tmp) / "configs")
        shutil.copytree(ROOT / "scripts", Path(tmp) / "scripts")
        probe = subprocess.run(
            [sys.executable, "-c", "import hddlab; print(hddlab.__file__)"],
            env={**os.environ, "PYTHONPATH": str(src)}, capture_output=True, text=True,
        )
        if not probe.stdout.strip().startswith(str(src)):
            print("The temporary copy is not the package being imported; aborting.", file=sys.stderr)
            return 2
        if not run_tests(src):
            print("The unmutated test suite fails; fix that first.", file=sys.stderr)
            return 2
        caught = 0
        for description, filename, original, mutated in MUTATIONS:
            path = src / "hddlab" / filename
            text = path.read_text()
            if text.count(original) != 1:
                print(f"SETUP ERROR  {description}: target text found {text.count(original)} times")
                return 2
            path.write_text(text.replace(original, mutated))
            passed = run_tests(src)
            path.write_text(text)
            caught += not passed
            print(f"{'caught ' if not passed else 'MISSED '}  {description}")
        print(f"\n{caught} of {len(MUTATIONS)} mutations caught.")
        return 0 if caught == len(MUTATIONS) else 1


if __name__ == "__main__":
    sys.exit(main())
