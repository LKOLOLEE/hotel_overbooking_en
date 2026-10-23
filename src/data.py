"""Loading, downloading and cleaning of the 'Hotel Booking Demand' dataset."""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], start=1)}

CATEGORICAL = [
    "hotel", "meal", "market_segment", "distribution_channel",
    "reserved_room_type", "deposit_type", "customer_type", "agent_cat",
]


def download_if_missing(path=C.DATA_RAW, url=C.DATA_URL) -> None:
    """Downloads the CSV if it is not available locally."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading data from {url} ...")
    try:
        urllib.request.urlretrieve(url, path)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Could not download the data. Download 'hotel_bookings.csv' "
            "from Kaggle (jessemostipak/hotel-booking-demand) and save it as "
            f"{path}"
        ) from exc


def load_raw(path=C.DATA_RAW) -> pd.DataFrame:
    path = Path(path)
    if path.resolve() == C.DATA_RAW.resolve():
        download_if_missing(path)          # demo data is downloaded automatically
    elif not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Cleaning and derived columns (dates, length of stay, target)."""
    df = df.copy()
    df["booking_id"] = np.arange(len(df))

    df["arrival"] = pd.to_datetime(dict(
        year=df["arrival_date_year"],
        month=df["arrival_date_month"].map(MONTHS),
        day=df["arrival_date_day_of_month"],
    ))
    df["booking_date"] = df["arrival"] - pd.to_timedelta(df["lead_time"], unit="D")
    df["status_date"] = pd.to_datetime(df["reservation_status_date"])
    df["nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["departure"] = df["arrival"] + pd.to_timedelta(df["nights"], unit="D")

    # Target: the guest did NOT arrive (cancellation or no-show)
    df["no_arrival"] = df["reservation_status"].isin(["Canceled", "No-Show"]).astype(int)
    df["is_no_show"] = (df["reservation_status"] == "No-Show").astype(int)
    df["is_cancel"] = (df["reservation_status"] == "Canceled").astype(int)

    # How many days before arrival the cancellation happened (NaN for the rest)
    df["cancel_days_before"] = np.where(
        df["is_cancel"] == 1,
        (df["arrival"] - df["status_date"]).dt.days,
        np.nan,
    )

    # Missing values and outliers
    df["children"] = df["children"].fillna(0).astype(int)
    df = df[(df["adults"] + df["children"] + df["babies"]) > 0]
    df = df[df["nights"] > 0]                       # bookings without an overnight stay
    df["adr"] = df["adr"].clip(lower=0)
    df = df[df["adr"] <= 1000]                      # single data error of EUR 5,400

    # Agent / company as categories
    agent = df["agent"].fillna(-1).astype(int)
    top = agent[agent >= 0].value_counts().head(C.TOP_AGENTS).index
    df["agent_cat"] = np.where(agent.isin(top), "A" + agent.astype(str),
                               np.where(agent < 0, "none", "other"))
    df["has_company"] = df["company"].notna().astype(int)
    df["meal"] = df["meal"].replace({"Undefined": "SC"})

    for col in CATEGORICAL:
        df[col] = df[col].astype(str)

    df["revenue"] = df["adr"] * df["nights"]
    return df.reset_index(drop=True)


def time_split(arrival: pd.Series) -> pd.Series:
    """Split label based on the arrival date."""
    return pd.Series(
        np.select(
            [arrival <= C.TRAIN_END, arrival <= C.VALID_END],
            ["train", "valid"], default="test"),
        index=arrival.index)


def load_clean(path=None) -> pd.DataFrame:
    """path: alternative CSV file (e.g. a daily PMS export in the same format)."""
    df = clean(load_raw(path or C.DATA_RAW))
    df["split"] = time_split(df["arrival"])
    return df
