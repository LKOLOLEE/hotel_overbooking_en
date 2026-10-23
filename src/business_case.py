"""Business case: annual financial impact, ROI and payback period."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
from src.overbooking import POLICY_NAMES
from src.plotting import COL, eur_axis, titles


def build(bt_summary: pd.DataFrame, bt_nights: int, outreach_annual: float,
          policy_annual: float) -> tuple[pd.DataFrame, dict]:
    factor = 365 / bt_nights
    ob_annual = bt_summary.loc[POLICY_NAMES["model"], "Net result [EUR]"] * factor
    oracle_annual = bt_summary.loc[POLICY_NAMES["oracle"], "Net result [EUR]"] * factor
    benefit = ob_annual + outreach_annual
    net_year = benefit - C.RUN_COST_EUR_PER_YEAR
    items = pd.DataFrame([
        {"Item": "Model-based overbooking (backtest, annualised)", "EUR / year": ob_annual,
         "Source": "public data + synthetic turned-away demand"},
        {"Item": "Preventive actions for high-risk bookings", "EUR / year": outreach_annual,
         "Source": "SYNTHETIC experiment"},
        {"Item": "Cost of running the solution", "EUR / year": -C.RUN_COST_EUR_PER_YEAR,
         "Source": "assumption"},
        {"Item": "TOTAL (excluding policy changes)", "EUR / year": net_year, "Source": ""},
        {"Item": "Additionally: potential of prepayment policy changes (to be validated by a test)",
         "EUR / year": policy_annual, "Source": "what-if simulation, SYNTHETIC assumptions"},
        {"Item": "For comparison: overbooking upper bound (oracle)",
         "EUR / year": oracle_annual, "Source": "backtest"},
    ]).set_index("Item")
    kpi = {
        "Net benefit per year [EUR]": net_year,
        "Project cost [EUR]": C.PROJECT_COST_EUR,
        "First-year ROI": (net_year - C.PROJECT_COST_EUR) / C.PROJECT_COST_EUR,
        "Payback period [months]": C.PROJECT_COST_EUR / (net_year / 12) if net_year > 0 else float("nan"),
        "Benefit per hotel per year [EUR]": net_year / 2,
    }
    return items, kpi


def fig_waterfall(items: pd.DataFrame):
    parts = items.iloc[:3]
    vals = parts["EUR / year"].to_numpy()
    labels = ["Overbooking", "Preventive\nactions", "Running cost"]
    start = 0.0
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, v in enumerate(vals):
        ax.bar(i, v, bottom=start, color=COL["good"] if v >= 0 else COL["critical"], width=0.6)
        ax.text(i, start + v + (abs(vals).max() * 0.03 if v >= 0 else -abs(vals).max() * 0.08),
                f"{'+' if v >= 0 else '-'}€{abs(v) / 1000:,.0f}k", ha="center", fontsize=9, color=COL["text"])
        start += v
    ax.bar(len(vals), start, color=COL["s1"], width=0.6)
    ax.text(len(vals), start + abs(vals).max() * 0.03, f"€{start / 1000:,.0f}k",
            ha="center", fontsize=9, fontweight="bold", color=COL["text"])
    ax.set_xticks(range(len(vals) + 1), labels + ["Total"])
    peak = max(np.cumsum(vals).max(), start)
    ax.set_ylim(min(0, np.cumsum(vals).min()) * 1.1, peak * 1.2)
    ax.axhline(0, color=COL["neutral"], lw=1)
    eur_axis(ax)
    ax.grid(axis="x", visible=False)
    titles(ax, "Annual financial impact (2 hotels)", "Excluding the potential of prepayment policy changes")
    return fig
