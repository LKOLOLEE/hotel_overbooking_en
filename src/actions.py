"""Daily list of high-risk bookings with reasons and the recommended action."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.eda import HOTEL_NAMES
from src.explain import risk_reasons

ACTION_PHONE = "Phone: confirm the stay + offer a date change / discounted prepayment"
ACTION_EMAIL = "E-mail: ask the guest to confirm arrival + prepayment offer"
ACTION_NONE = "No action"


def risk_list(df, model, as_of, days_ahead=14, contact_threshold=0.25, call_threshold=0.5):
    as_of = pd.Timestamp(as_of)
    dta = (df["arrival"] - as_of).dt.days
    active = (dta >= 0) & (dta <= days_ahead) & (df["lead_time"] >= dta) & (
        (df["is_cancel"] == 0) | (df["cancel_days_before"] <= dta))
    rows = df[active].copy()
    rows["dta"] = dta[active]
    rows["p"] = model.predict(rows, days_to_arrival=rows["dta"])
    rows["revenue_at_risk"] = rows["p"] * rows["revenue"]
    rows = rows.sort_values("p", ascending=False)

    def action(p):
        if p >= call_threshold:
            return ACTION_PHONE
        if not np.isnan(contact_threshold) and p >= contact_threshold:
            return ACTION_EMAIL
        return ACTION_NONE
    rows["action"] = rows["p"].map(action)
    top = rows[rows["action"] != ACTION_NONE].copy()
    top["reasons"] = risk_reasons(model, top, days_to_arrival=top["dta"]) if len(top) else []

    out = top.assign(hotel=top["hotel"].map(HOTEL_NAMES))[[
        "booking_id", "hotel", "arrival", "dta", "nights", "distribution_channel",
        "market_segment", "deposit_type", "adr", "revenue", "p", "revenue_at_risk",
        "action", "reasons", "no_arrival"]]
    out = out.rename(columns={
        "booking_id": "Booking ID (pseudonym)", "hotel": "Hotel", "arrival": "Arrival",
        "dta": "Days to arrival", "nights": "Nights", "distribution_channel": "Channel",
        "market_segment": "Segment", "deposit_type": "Deposit", "adr": "ADR [EUR]",
        "revenue": "Booking value [EUR]", "p": "No-arrival risk",
        "revenue_at_risk": "Revenue at risk [EUR]", "action": "Recommended action",
        "reasons": "Main risk drivers", "no_arrival": "Did not arrive (verification)"})
    summary = {
        "Active bookings (arrival within 14 days)": len(rows),
        "Phone contact": int((rows["action"] == ACTION_PHONE).sum()),
        "E-mail contact": int((rows["action"] == ACTION_EMAIL).sum()),
        "Expected no-arrivals": float(rows["p"].sum()),
        "Actual no-arrivals (verification)": int(rows["no_arrival"].sum()),
        "Revenue at risk [EUR]": float(rows["revenue_at_risk"].sum()),
    }
    return out.reset_index(drop=True), summary
