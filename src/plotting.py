"""Shared chart style (matplotlib) and saving to files."""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter

import config as C

COL = C.COLORS


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": COL["surface"],
        "axes.facecolor": COL["surface"],
        "savefig.facecolor": COL["surface"],
        "axes.edgecolor": COL["grid"],
        "axes.labelcolor": COL["text2"],
        "axes.titlecolor": COL["text"],
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": COL["grid"],
        "grid.linewidth": 0.6,
        "xtick.color": COL["text2"],
        "ytick.color": COL["text2"],
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
        "font.family": "DejaVu Sans",
        "figure.dpi": 110,
        "savefig.dpi": 140,
        "savefig.bbox": "tight",
        "figure.max_open_warning": 100,
    })


def pct_axis(ax, axis="y", decimals=0):
    fmt = PercentFormatter(1.0, decimals=decimals)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def eur_axis(ax, axis="y"):
    def f(x, _):
        sign = "-" if x < 0 else ""
        x = abs(x)
        if x >= 1e6:
            return f"{sign}€{x / 1e6:.1f}m"
        if x >= 1e3:
            return f"{sign}€{x / 1e3:.0f}k"
        return f"{sign}€{x:.0f}"
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(FuncFormatter(f))


def titles(ax, title: str, sub: str | None = None):
    """Bold title + subtitle carrying the takeaway, left-aligned."""
    ax.set_title(title, pad=22 if sub else 8)
    if sub:
        ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=9,
                color=COL["text2"], va="bottom", ha="left")


def save(fig, name: str):
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    path = C.FIGURES / f"{name}.png"
    fig.savefig(path)
    return path


apply_style()
