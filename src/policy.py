"""
Segment analysis and cancellation / deposit policy recommendations.

"What-if" simulation of a prepayment requirement in a segment [SYNTHETIC assumptions]:
  * the cancellation rate falls by POLICY_CANCEL_REDUCTION (relative)
  * the number of bookings falls by POLICY_BOOKING_LOSS (some guests walk away)
Annual value = saved cancellations x value (late: no resale,
early: resold with probability POLICY_EARLY_RESALE)
             - lost bookings x value x (1 - POLICY_REPLACEMENT)
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
from src.eda import HOTEL_NAMES, LEAD_BINS, LEAD_LABELS
from src.plotting import COL, eur_axis, pct_axis, titles


def segment_table(df: pd.DataFrame) -> pd.DataFrame:
    d = df.assign(lead_bucket=pd.cut(df["lead_time"], LEAD_BINS, labels=LEAD_LABELS).astype(str))
    d["late_cancel"] = ((d["is_cancel"] == 1) & (d["cancel_days_before"] <= 7)).astype(int)
    keys = ["hotel", "distribution_channel", "deposit_type", "lead_bucket"]
    g = d.groupby(keys).agg(
        bookings=("booking_id", "size"),
        cancel_rate=("is_cancel", "mean"),
        late_cancel_rate=("late_cancel", "mean"),
        no_show=("is_no_show", "mean"),
        avg_revenue=("revenue", "mean"),
    ).reset_index()
    g["revenue_lost"] = g["bookings"] * (g["cancel_rate"] + g["no_show"]) * g["avg_revenue"]
    g["revenue_lost_late"] = g["bookings"] * (g["late_cancel_rate"] + g["no_show"]) * g["avg_revenue"]
    return g


def whatif_prepayment(seg: pd.DataFrame, years: float) -> pd.DataFrame:
    s = seg[(seg["deposit_type"] == "No Deposit") & (seg["bookings"] >= C.POLICY_MIN_BOOKINGS)].copy()
    red, loss = C.POLICY_CANCEL_REDUCTION, C.POLICY_BOOKING_LOSS
    B, c, late = s["bookings"], s["cancel_rate"], s["late_cancel_rate"]
    saved_late = B * (1 - loss) * late * red
    saved_early = B * (1 - loss) * (c - late) * red
    lost = B * loss * (1 - c)                         # walked away, but would have arrived
    s["stays_change"] = saved_late + saved_early - lost
    value = (saved_late + saved_early * (1 - C.POLICY_EARLY_RESALE)
             - lost * (1 - C.POLICY_REPLACEMENT)) * s["avg_revenue"]
    s["revenue_change_per_year"] = value / years
    # less uncertainty = fewer empty rooms caused by late cancellations
    s["fewer_late_cancels_per_year"] = s["bookings"] * s["late_cancel_rate"] * red / years
    s["recommendation"] = np.where(
        s["revenue_change_per_year"] > 0,
        "Introduce prepayment / shorter free-cancellation window",
        "Keep current terms (cost of lost bookings > benefit)")
    out = s.sort_values("revenue_change_per_year", ascending=False)
    out["hotel"] = out["hotel"].map(HOTEL_NAMES)
    return out.rename(columns={
        "hotel": "Hotel", "distribution_channel": "Channel", "deposit_type": "Deposit",
        "lead_bucket": "Lead time (days)", "bookings": "Bookings",
        "cancel_rate": "Cancellation rate", "late_cancel_rate": "Late cancellation rate",
        "no_show": "No-show rate", "avg_revenue": "Avg. booking revenue [EUR]",
        "revenue_lost": "Revenue cancelled [EUR]",
        "revenue_lost_late": "Revenue lost late [EUR]",
        "stays_change": "Change in completed stays",
        "revenue_change_per_year": "Revenue change per year [EUR]",
        "fewer_late_cancels_per_year": "Fewer late cancellations per year",
        "recommendation": "Recommendation"})


def channel_table(df: pd.DataFrame, years: float) -> pd.DataFrame:
    g = df.groupby(["hotel", "distribution_channel"]).agg(
        bookings=("booking_id", "size"),
        cancel_rate=("is_cancel", "mean"),
        no_show=("is_no_show", "mean"),
        revenue_cancelled=("revenue", lambda r: r[df.loc[r.index, "is_cancel"] == 1].sum()),
    ).reset_index()
    g = g[g["bookings"] >= 100]
    g["revenue_cancelled"] /= years
    g["hotel"] = g["hotel"].map(HOTEL_NAMES)
    return g.rename(columns={"hotel": "Hotel", "distribution_channel": "Channel",
                             "bookings": "Bookings", "cancel_rate": "Cancellation rate",
                             "no_show": "No-show rate",
                             "revenue_cancelled": "Revenue cancelled per year [EUR]"})


def fig_channel_heat(df):
    d = df.assign(lb=pd.cut(df["lead_time"], LEAD_BINS, labels=LEAD_LABELS))
    t = d.groupby(["distribution_channel", "lb"], observed=True)["no_arrival"].agg(["mean", "size"])
    t = t[t["size"] >= 100]["mean"].unstack()
    t = t.loc[[c for c in ["Direct", "Corporate", "TA/TO", "GDS"] if c in t.index]]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", C.COLORS["seq"])
    fig, ax = plt.subplots(figsize=(9, 3.4))
    im = ax.imshow(t.values, cmap=cmap, vmin=0, vmax=max(0.6, np.nanmax(t.values)), aspect="auto")
    ax.set_xticks(range(t.shape[1]), t.columns)
    ax.set_yticks(range(t.shape[0]), t.index)
    ax.grid(False)
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            v = t.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=9,
                        color="white" if v > 0.35 else COL["text"])
    ax.set_xlabel("Booking lead time (days)")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    titles(ax, "Risk map: channel x lead time",
           "Share of bookings without arrival (cells with at least 100 bookings)")
    return fig


def fig_whatif(wi, top=10):
    s = wi.head(top).iloc[::-1]
    labels = [f"{h} | {k} | {w} days" for h, k, w in zip(s["Hotel"], s["Channel"], s["Lead time (days)"])]
    vals = s["Revenue change per year [EUR]"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(labels, vals, color=[COL["s1"] if v > 0 else COL["critical"] for v in vals], height=0.6)
    ax.axvline(0, color=COL["neutral"], lw=1)
    eur_axis(ax, "x")
    ax.grid(axis="y", visible=False)
    titles(ax, "Where prepayment pays off (what-if simulation)",
           f"Annual net revenue change; assumptions: cancellations -{C.POLICY_CANCEL_REDUCTION:.0%}, "
           f"bookings -{C.POLICY_BOOKING_LOSS:.0%}, resale of early cancellations {C.POLICY_EARLY_RESALE:.0%} [SYNTHETIC]")
    return fig
