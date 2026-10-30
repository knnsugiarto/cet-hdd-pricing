"""End-to-end smoke test of the study and the forecast script on synthetic data."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from hddlab.config import REPO_ROOT
from hddlab.pipeline import run_study
from hddlab.report import README_END, README_START
from hddlab.synthetic import synthetic_cet, to_hadcet_text


def test_study_runs_and_freezes_before_holdout(small_cfg, temps: pd.Series, tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{README_START}\nplaceholder\n{README_END}\noutro\n")
    res = run_study(small_cfg, temps, tmp_path / "results", {}, None, readme, make_figures=True)

    frozen = json.loads((tmp_path / "results" / "frozen.json").read_text())["frozen"]
    assert set(frozen) == {"burn", "index", "daily"}
    # The holdout contains only frozen variants and the baseline.
    assert set(res.holdout["variant"]) == set(frozen.values()) | {small_cfg.evaluation.baseline}
    assert len(res.verdicts) == 3
    text = readme.read_text()
    assert text.startswith("intro") and text.rstrip().endswith("outro")
    assert "placeholder" not in text and "Pre-registered predictions" in text
    for name in ("seasonal_index", "walkforward_intervals", "pit_development", "closed_form_check", "coverage"):
        assert (tmp_path / "results" / "figures" / f"{name}.png").exists()
    assert (tmp_path / "results" / "holdout_runs.log").read_text().count("\n") == 1


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_forecast_make_and_score(small_cfg, tmp_path: Path) -> None:
    cfg = dataclasses.replace(
        small_cfg,
        root=tmp_path,
        forecast_dir=tmp_path / "forecasts",
        data=dataclasses.replace(small_cfg.data, raw_path=tmp_path / "data" / "raw" / "daily.txt"),
    )
    cfg.data.raw_path.parent.mkdir(parents=True)
    cfg.data.raw_path.write_text(to_hadcet_text(synthetic_cet()))
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "frozen.json").write_text(
        json.dumps(
            {
                "frozen": {
                    "burn": "burn|trend=linear|window=30",
                    "index": "index|trend=none|window=30",
                    "daily": "daily|ar=1|harmonics=2|vol=constant|window=30",
                },
                "baseline": "burn|trend=none|window=30",
            }
        )
    )
    forecast = _load_script("forecast")

    # A season that has not started: allowed. Scoring it must wait for data.
    path = forecast.make(cfg, "2026-12-01", "2027-03-31", allow_late=False)
    doc = json.loads(path.read_text())
    assert doc["data"]["last_date_used"] == "2026-10-27"
    assert set(doc["methods"]) == {"baseline", "burn", "index", "daily"}
    with pytest.raises(SystemExit):
        forecast.score(cfg, path)

    # A season inside the data is refused unless explicitly allowed, then scores.
    with pytest.raises(SystemExit):
        forecast.make(cfg, "2025-12-01", "2026-03-31", allow_late=False)
    late = forecast.make(cfg, "2025-12-01", "2026-03-31", allow_late=True)
    assert json.loads(late.read_text())["data"]["last_date_used"] == "2025-11-30"
    forecast.score(cfg, late)
    scored = json.loads(late.with_name(late.stem + "_score.json").read_text())
    assert set(scored["methods"]) == {"baseline", "burn", "index", "daily"}
