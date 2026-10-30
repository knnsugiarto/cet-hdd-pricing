"""Static figures for the README and results folder (matplotlib, PNG).

Colour follows the method, never its rank: burn blue, index orange, daily aqua,
the baseline grey and dashed, observations near-black.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

from hddlab.report import short_label  # noqa: E402

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
BASELINE = "#8a8985"
COLOURS = {"burn": "#2a78d6", "index": "#eb6834", "daily": "#1baf7a"}


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK_2, labelsize=8, width=0.6)
    ax.yaxis.label.set_color(INK_2)
    ax.xaxis.label.set_color(INK_2)


def _figure(nrows: int, ncols: int, size: tuple[float, float], **kw) -> tuple:
    fig, axes = plt.subplots(nrows, ncols, figsize=size, facecolor=SURFACE, **kw)
    for ax in np.atleast_1d(axes).ravel():
        _style(ax)
    return fig, axes


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def seasonal_index(
    h: pd.Series, dev_years: list[int], holdout_years: list[int], base: float, path: Path
) -> None:
    """Realised H by season, with the development and holdout periods marked."""
    fig, ax = _figure(1, 1, (8.0, 3.2))
    ax.axvspan(dev_years[0] - 0.5, dev_years[-1] + 0.5, color="#eef4fc", lw=0)
    ax.axvspan(holdout_years[0] - 0.5, holdout_years[-1] + 0.5, color="#fdf0ea", lw=0)
    ax.plot(h.index, h.to_numpy(), color=INK, lw=1.0)
    ax.plot(h.index, h.to_numpy(), "o", color=INK, ms=2.5)
    top = ax.get_ylim()[1]
    ax.text(dev_years[0], top, " development", va="top", fontsize=8, color=INK_2)
    ax.text(holdout_years[0], top, " holdout", va="top", fontsize=8, color=INK_2)
    ax.set_ylabel(f"Season HDD total (base {base:g} °C)")
    ax.set_xlabel("Season start year (Nov to Mar)")
    ax.set_title("Winter heating degree days, Central England Temperature", fontsize=10, color=INK, loc="left")
    _save(fig, path)


def walkforward_intervals(
    frame: pd.DataFrame, frozen: dict[str, str], holdout_first: int, base: float, path: Path
) -> None:
    """Each frozen method's forecast mean and 90% interval against the realised index."""
    fig, axes = _figure(3, 1, (8.0, 7.2), sharex=True)
    for ax, fam in zip(axes, ("burn", "index", "daily")):
        g = frame[(frame["variant"] == frozen[fam]) & (frame["base"] == base)].sort_values("year")
        c = COLOURS[fam]
        ax.fill_between(g["year"], g["lo90"], g["hi90"], color=c, alpha=0.18, lw=0)
        ax.plot(g["year"], g["mean"], color=c, lw=1.4)
        ax.plot(g["year"], g["realised"], "o", color=INK, ms=3)
        ax.axvline(holdout_first - 0.5, color=INK_2, lw=0.8, ls=":")
        ax.set_title(short_label(frozen[fam]), fontsize=9, color=INK, loc="left")
        ax.set_ylabel("HDD")
    axes[-1].set_xlabel("Season start year (dotted line: holdout starts)")
    handles = [
        plt.Line2D([], [], color=INK_2, lw=1.4, label="forecast mean"),
        plt.Rectangle((0, 0), 1, 1, color=INK_2, alpha=0.18, label="90% interval"),
        plt.Line2D([], [], color=INK, marker="o", ls="", ms=3, label="realised"),
    ]
    axes[0].legend(handles=handles, fontsize=8, frameon=False, ncol=3, loc="upper right")
    _save(fig, path)


def pit_histograms(
    frame: pd.DataFrame, variants: dict[str, str], base: float, path: Path, bins: int = 10
) -> None:
    """PIT histograms on development seasons; a calibrated forecast gives a flat histogram."""
    fig, axes = _figure(1, len(variants), (2.3 * len(variants), 2.6), sharey=True)
    for ax, (name, label) in zip(np.atleast_1d(axes), variants.items()):
        g = frame[(frame["variant"] == label) & (frame["base"] == base)]
        colour = COLOURS.get(name, BASELINE)
        ax.hist(g["pit"], bins=bins, range=(0, 1), color=colour, edgecolor=SURFACE, linewidth=1.5)
        ax.axhline(len(g) / bins, color=INK_2, lw=0.8, ls="--")
        ax.set_title(name, fontsize=9, color=INK, loc="left")
        ax.set_xlabel("PIT")
    np.atleast_1d(axes)[0].set_ylabel("Seasons")
    _save(fig, path)


def closed_form_check(frame: pd.DataFrame, base: float, path: Path) -> None:
    """Monte Carlo minus closed form, in standard errors, pooled over daily-model rows."""
    daily = frame[(frame["family"] == "daily") & (frame["base"] == base)]
    fig, axes = _figure(1, 2, (8.0, 2.8), sharey=True)
    grid = np.linspace(-4, 4, 200)
    panels = (
        ("z_call_l", "ATM call on the linear index L vs Bachelier"),
        ("z_mean_h", "Mean of H vs exact E[H]"),
    )
    for ax, (col, title) in zip(axes, panels):
        z = daily[col].to_numpy()
        ax.hist(z, bins=24, range=(-4, 4), density=True, color=COLOURS["daily"], edgecolor=SURFACE, linewidth=1.0)
        ax.plot(grid, norm.pdf(grid), color=INK, lw=1.2)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.set_xlabel("(Monte Carlo - exact) / MC standard error")
    axes[0].set_ylabel("Density (black: N(0, 1))")
    _save(fig, path)


def coverage_plot(
    dev_table: pd.DataFrame,
    holdout_table: pd.DataFrame,
    variants: dict[str, str],
    levels: tuple[float, ...],
    base: float,
    path: Path,
) -> None:
    """Empirical vs nominal interval coverage; points below the diagonal mean too-narrow intervals."""
    fig, axes = _figure(1, 2, (8.0, 3.2), sharey=True)
    for ax, (title, table) in zip(axes, (("Development", dev_table), ("Holdout", holdout_table))):
        ax.plot([0.4, 1.0], [0.4, 1.0], color=INK_2, lw=0.8, ls="--")
        offsets = np.linspace(-0.012, 0.012, len(variants))
        for off, (name, label) in zip(offsets, variants.items()):
            row = table[(table["variant"] == label) & (table["base"] == base)]
            if row.empty:
                continue
            ys = [float(row[f"cov{int(l * 100)}"].iloc[0]) for l in levels]
            ax.plot(np.array(levels) + off, ys, "o", ms=6, color=COLOURS.get(name, BASELINE), label=name)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.set_xlabel("Nominal coverage")
        ax.set_xlim(0.4, 1.0)
        ax.set_ylim(0.2, 1.02)
    axes[0].set_ylabel("Empirical coverage")
    axes[1].legend(fontsize=8, frameon=False, loc="lower right")
    _save(fig, path)


def risk_loading(prices: pd.DataFrame, path: Path) -> None:
    """Price in pounds against the risk-loading alpha, one line per method."""
    fig, ax = _figure(1, 1, (5.5, 3.2))
    order = [f for f in ("baseline", "burn", "index", "daily") if f in set(prices["family"])]
    for family in order:
        g = prices[prices["family"] == family].sort_values("alpha")
        ax.plot(g["alpha"], g["price_gbp"], "o-", color=COLOURS.get(family, BASELINE), lw=1.4, ms=6, label=family)
    ax.set_xlabel("Risk loading alpha (price = E[payoff] + alpha sd[payoff])")
    ax.set_ylabel("ATM call price (GBP, undiscounted)")
    ax.legend(fontsize=8, frameon=False)
    _save(fig, path)
