"""
Building the leakage-free "snapshot" dataset and the model features.

Idea: the model predicts risk at a specific moment before arrival (D-h).
For every booking and every horizon h we create a row only if the booking was
ACTIVE on day D-h:
  * it had already been made:      lead_time >= h
  * it had not yet been cancelled: the cancellation happened on D-h or later
Target: whether the guest ultimately did not arrive (cancellation after D-h or no-show).
Only information known on day D-h is used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
from src.data import CATEGORICAL

# Note: arrival month/week and price (ADR) were deliberately EXCLUDED from the
# model. With ~1 year of training history the model "memorised" the season of
# a specific year, and prices drift (inflation, rate changes) - in the backtest
# this lowered ROC AUC on the test period (see README).
NUMERIC_FEATURES = [
    "lead_time", "days_to_arrival", "days_since_booking", "arrival_dow",
    "stays_in_weekend_nights", "stays_in_week_nights", "nights",
    "adults", "children", "babies", "is_repeated_guest",
    "previous_cancellations", "previous_bookings_not_canceled",
    "has_company", "days_in_waiting_list",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL

FEATURE_LABELS = {
    "lead_time": "Booking lead time (days)",
    "days_to_arrival": "Days to arrival",
    "days_since_booking": "Days since booking",
    "arrival_month": "Arrival month",
    "arrival_week": "Arrival week",
    "arrival_dow": "Arrival day of week",
    "stays_in_weekend_nights": "Weekend nights",
    "stays_in_week_nights": "Weekday nights",
    "nights": "Length of stay",
    "adults": "Adults",
    "children": "Children",
    "babies": "Babies",
    "is_repeated_guest": "Repeat guest",
    "previous_cancellations": "Guest's previous cancellations",
    "previous_bookings_not_canceled": "Guest's previous completed stays",
    "has_company": "Company booking",
    "days_in_waiting_list": "Days on waiting list",
    "adr": "Rate per night (ADR)",
    "adr_per_person": "Rate per night per person",
    "hotel": "Hotel",
    "meal": "Meal plan",
    "market_segment": "Market segment",
    "distribution_channel": "Distribution channel",
    "reserved_room_type": "Room type",
    "deposit_type": "Deposit type",
    "customer_type": "Customer type",
    "agent_cat": "Agent / intermediary",
}


def snapshot_mask(df: pd.DataFrame, h: int) -> pd.Series:
    """Whether the booking was active h days before arrival."""
    booked = df["lead_time"] >= h
    not_yet_cancelled = (df["is_cancel"] == 0) | (df["cancel_days_before"] <= h)
    return booked & not_yet_cancelled


def build_snapshots(df: pd.DataFrame, horizons=None) -> pd.DataFrame:
    horizons = horizons or C.SNAPSHOT_HORIZONS
    parts = []
    for h in horizons:
        part = df.loc[snapshot_mask(df, h)].copy()
        part["days_to_arrival"] = h
        parts.append(part)
    snap = pd.concat(parts, ignore_index=True)
    snap["snapshot_date"] = snap["arrival"] - pd.to_timedelta(snap["days_to_arrival"], unit="D")
    return snap


def make_features(rows: pd.DataFrame, days_to_arrival=None,
                  categories: dict | None = None) -> pd.DataFrame:
    """Feature matrix for rows (bookings) at a given moment before arrival.

    days_to_arrival: Series/array; if None, the rows['days_to_arrival'] column is used.
    categories: dictionary of category levels from training (consistency at scoring time).
    """
    X = pd.DataFrame(index=rows.index)
    dta = rows["days_to_arrival"] if days_to_arrival is None else days_to_arrival
    X["days_to_arrival"] = np.asarray(dta, dtype=float)
    X["lead_time"] = rows["lead_time"].astype(float)
    X["days_since_booking"] = X["lead_time"] - X["days_to_arrival"]
    X["arrival_dow"] = rows["arrival"].dt.dayofweek
    for col in ["stays_in_weekend_nights", "stays_in_week_nights", "nights",
                "adults", "children", "babies", "is_repeated_guest",
                "previous_cancellations", "previous_bookings_not_canceled",
                "has_company"]:
        X[col] = rows[col].astype(float)
    # Days on the waiting list are only known up to the snapshot moment
    X["days_in_waiting_list"] = np.minimum(rows["days_in_waiting_list"],
                                           X["days_since_booking"].clip(lower=0))
    for col in CATEGORICAL:
        if categories is not None:
            vals = rows[col].astype(str)
            vals = vals.where(vals.isin(categories[col]))      # unknown -> missing
            X[col] = pd.Categorical(vals, categories=categories[col])
        else:
            X[col] = rows[col].astype("category")
    return X[FEATURES]


def category_levels(rows: pd.DataFrame) -> dict:
    return {col: sorted(rows[col].astype(str).unique().tolist()) for col in CATEGORICAL}
