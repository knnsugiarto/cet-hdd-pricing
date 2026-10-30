"""Shared fixtures: synthetic CET-like data and a small, fast configuration."""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from hddlab.config import Config, load_config
from hddlab.synthetic import synthetic_cet


@pytest.fixture(scope="session")
def cfg() -> Config:
    return load_config()


@pytest.fixture(scope="session")
def small_cfg(cfg: Config) -> Config:
    """Few seasons and paths, so pipeline tests run in seconds."""
    return dataclasses.replace(
        cfg,
        evaluation=dataclasses.replace(
            cfg.evaluation, dev_first=1950, dev_last=1955, holdout_first=1956, holdout_last=1958
        ),
        daily=dataclasses.replace(cfg.daily, n_paths=400),
    )


@pytest.fixture(scope="session")
def temps() -> pd.Series:
    """Synthetic daily temperatures from the configured start date to 31 Mar 2026."""
    return synthetic_cet().loc["1900-11-01":"2026-03-31"]
