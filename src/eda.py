"""Exploratory analysis: scale of the cancellation and no-show problem."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
from src.plotting import COL, pct_axis, titles

HOTEL_NAMES = {"City Hotel": "City hotel", "Resort Hotel": "Resort hotel"}
LEAD_BINS = [-1, 7, 30, 90, 180, 365, 10_000]
LEAD_LABELS = ["0-7", "8-30", "31-90", "91-180", "181-365", "365+"]


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for hotel, g in df.groupby("hotel"):
        canc = g[g["is_cancel"] == 1]
        late = canc[canc["cancel_days_before"] <= 7]
        rows[HOTEL_NAMES[hotel]] = {
            "Bookings": len(g),
            "Cancellation rate": g["is_cancel"].mean(),
            "No-show rate": g["is_no_show"].mean(),
            "Share of cancellations made in the last 7 days": len(late) / max(len(canc), 1),
            "Avg. booking lead time (days)": g["lead_time"].mean(),
            "Avg. rate per night (ADR) [EUR]": g.loc[g["no_arrival"] == 0, "adr"].mean(),
            "Revenue cancelled late (<=7 days) [EUR]": late["revenue"].sum(),
            "Revenue lost to no-shows [EUR]": g.loc[g["is_no_show"] == 1, "revenue"].sum(),
        }
    return pd.DataFrame(rows).T          # rows = hotels


def fig_cancel_trend(df):
    m = (df.assign(month=df["arrival"].dt.to_period("M").dt.to_timestamp())
           .groupby(["hotel", "month"])["no_arrival"].mean().reset_index())
    fig, ax = plt.subplots(figsize=(9, 4))
    for hotel, g in m.groupby("hotel"):
        ax.plot(g["month"], g["no_arrival"], color=C.HOTEL_COLORS[hotel], label=HOTEL_NAMES[hotel])
    pct_axis(ax)
    ax.set_ylim(0, None)
    ax.set_ylabel("Share of bookings without arrival")
    ax.legend(loc="lower left")
    titles(ax, "One in three bookings never turns into a stay",
           "Cancellation and no-show rate by arrival month")
    return fig


def fig_cancel_by_lead(df):
    d = df.assign(lb=pd.cut(df["lead_time"], LEAD_BINS, labels=LEAD_LABELS))
    t = d.groupby(["lb", "hotel"], observed=True)["no_arrival"].mean().unstack()
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(t))
    w = 0.38
    for i, hotel in enumerate(t.columns):
        ax.bar(x + (i - 0.5) * w, t[hotel], width=w - 0.02, color=C.HOTEL_COLORS[hotel],
               label=HOTEL_NAMES[hotel])
    ax.set_xticks(x, t.index)
    ax.set_xlabel("Booking lead time (days)")
    pct_axis(ax)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    titles(ax, "The earlier a booking is made, the higher the cancellation risk",
           "Share of bookings without arrival by lead time")
    return fig


def fig_cancel_by_segment(df):
    t = df.groupby(["market_segment", "hotel"])["no_arrival"].agg(["mean", "size"]).reset_index()
    t = t[t["size"] >= 200]
    order = t.groupby("market_segment")["mean"].mean().sort_values().index
    piv = t.pivot(index="market_segment", columns="hotel", values="mean").reindex(order)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    y = np.arange(len(piv))
    h = 0.38
    for i, hotel in enumerate(piv.columns):
        ax.barh(y + (i - 0.5) * h, piv[hotel], height=h - 0.03,
                color=C.HOTEL_COLORS[hotel], label=HOTEL_NAMES[hotel])
    ax.set_yticks(y, piv.index)
    pct_axis(ax, "x")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    titles(ax, "Groups and travel agents (TA/TO) cancel most often",
           "Share of bookings without arrival by market segment (min. 200 bookings)")
    return fig


def fig_cancel_timing(df):
    canc = df[df["is_cancel"] == 1]
    fig, ax = plt.subplots(figsize=(9, 4))
    grid = np.arange(0, 181)
    for hotel, g in canc.groupby("hotel"):
        d = g["cancel_days_before"].to_numpy()
        share = [(d <= x).mean() for x in grid]
        ax.plot(grid, share, color=C.HOTEL_COLORS[hotel], label=HOTEL_NAMES[hotel])
    ax.axvline(7, color=COL["neutral"], lw=1, ls="--")
    ax.text(8, 0.02, "7 days", color=COL["text2"], fontsize=9)
    pct_axis(ax)
    ax.set_xlabel("Days before arrival")
    ax.set_ylabel("Cumulative share of cancellations")
    ax.legend(loc="lower right")
    titles(ax, "Some cancellations arrive at the last minute - the room is then hard to resell",
           "Cumulative share of cancellations made within the last X days before arrival")
    return fig


def fig_occupancy(occ, capacity):
    hotels = sorted(capacity)
    fig, axes = plt.subplots(len(hotels), 1, figsize=(10, 5.5), sharex=True)
    for ax, hotel in zip(axes, hotels):
        g = occ[occ["hotel"] == hotel]
        ax.plot(g["night"], g["occupied"], color=C.HOTEL_COLORS[hotel], lw=1.2)
        ax.axhline(capacity[hotel], color=COL["critical"], lw=1, ls="--")
        ax.text(g["night"].iloc[3], capacity[hotel] + 3,
                f"Estimated capacity: {capacity[hotel]} rooms", fontsize=9, color=COL["text2"])
        ax.set_ylabel("Occupied rooms")
        ax.set_ylim(0, capacity[hotel] * 1.15)
        ax.set_title(HOTEL_NAMES[hotel], fontsize=10, pad=4)
    fig.suptitle("Occupancy regularly hits the ceiling - on those nights every empty room is lost revenue",
                 x=0.06, ha="left", fontsize=12, fontweight="bold", color=COL["text"])
    fig.tight_layout()
    return fig
