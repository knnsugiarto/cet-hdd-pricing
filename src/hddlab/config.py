"""Load the TOML configuration into typed, immutable objects."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.toml"


@dataclass(frozen=True)
class DataConfig:
    url: str
    raw_path: Path
    meta_path: Path
    dataset_version: str
    start_date: str
    provisional_from: str
    evaluation_end: str


@dataclass(frozen=True)
class ContractConfig:
    base_temp: float
    season_start: tuple[int, int]
    season_end: tuple[int, int]
    tick_gbp: float
    strike_reference_seasons: int
    strike_sd_multiples: tuple[float, ...]
    discount_rate: float
    loadings: tuple[float, ...]


@dataclass(frozen=True)
class EvaluationConfig:
    dev_first: int
    dev_last: int
    holdout_first: int
    holdout_last: int
    newey_west_lags: int
    intervals: tuple[float, ...]
    baseline: str

    @property
    def dev_years(self) -> list[int]:
        return list(range(self.dev_first, self.dev_last + 1))

    @property
    def holdout_years(self) -> list[int]:
        return list(range(self.holdout_first, self.holdout_last + 1))


@dataclass(frozen=True)
class SeasonalTotalsConfig:
    """Grid for the burn and index methods."""

    windows: tuple[int, ...]
    trends: tuple[str, ...]


@dataclass(frozen=True)
class DailyConfig:
    windows: tuple[int, ...]
    harmonics: tuple[int, ...]
    ar_orders: tuple[int, ...]
    vol: tuple[str, ...]
    vol_harmonics: int
    n_paths: int
    seed: int
    acf_lags: int


@dataclass(frozen=True)
class Config:
    data: DataConfig
    contract: ContractConfig
    evaluation: EvaluationConfig
    burn: SeasonalTotalsConfig
    index: SeasonalTotalsConfig
    daily: DailyConfig
    robustness_bases: tuple[float, ...]
    forecast_dir: Path
    root: Path

    @property
    def all_bases(self) -> tuple[float, ...]:
        """Headline base temperature first, then robustness bases."""
        return (self.contract.base_temp, *self.robustness_bases)


def load_config(path: Path | str = DEFAULT_CONFIG, root: Path | None = None) -> Config:
    """Read a TOML config. Relative paths in it resolve against ``root`` (default: repo root)."""
    path = Path(path)
    root = REPO_ROOT if root is None else Path(root)
    with path.open("rb") as fh:
        raw = tomllib.load(fh)

    d, c, e = raw["data"], raw["contract"], raw["evaluation"]
    data = DataConfig(
        url=d["url"],
        raw_path=root / d["raw_path"],
        meta_path=root / d["meta_path"],
        dataset_version=d["dataset_version"],
        start_date=d["start_date"],
        provisional_from=d["provisional_from"],
        evaluation_end=d["evaluation_end"],
    )
    contract = ContractConfig(
        base_temp=float(c["base_temp"]),
        season_start=(int(c["season_start"][0]), int(c["season_start"][1])),
        season_end=(int(c["season_end"][0]), int(c["season_end"][1])),
        tick_gbp=float(c["tick_gbp"]),
        strike_reference_seasons=int(c["strike_reference_seasons"]),
        strike_sd_multiples=tuple(float(x) for x in c["strike_sd_multiples"]),
        discount_rate=float(c["discount_rate"]),
        loadings=tuple(float(x) for x in c["loadings"]),
    )
    evaluation = EvaluationConfig(
        dev_first=int(e["dev_first"]),
        dev_last=int(e["dev_last"]),
        holdout_first=int(e["holdout_first"]),
        holdout_last=int(e["holdout_last"]),
        newey_west_lags=int(e["newey_west_lags"]),
        intervals=tuple(float(x) for x in e["intervals"]),
        baseline=e["baseline"],
    )
    burn = SeasonalTotalsConfig(tuple(raw["burn"]["windows"]), tuple(raw["burn"]["trends"]))
    index = SeasonalTotalsConfig(tuple(raw["index"]["windows"]), tuple(raw["index"]["trends"]))
    dd = raw["daily"]
    daily = DailyConfig(
        windows=tuple(dd["windows"]),
        harmonics=tuple(dd["harmonics"]),
        ar_orders=tuple(dd["ar_orders"]),
        vol=tuple(dd["vol"]),
        vol_harmonics=int(dd["vol_harmonics"]),
        n_paths=int(dd["n_paths"]),
        seed=int(dd["seed"]),
        acf_lags=int(dd["acf_lags"]),
    )
    return Config(
        data=data,
        contract=contract,
        evaluation=evaluation,
        burn=burn,
        index=index,
        daily=daily,
        robustness_bases=tuple(float(x) for x in raw["robustness"]["base_temps"]),
        forecast_dir=root / raw["forecast"]["output_dir"],
        root=root,
    )
