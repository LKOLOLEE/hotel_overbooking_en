"""
Forecast of actual occupancy and overbooking optimisation.

Two forecasting stages for night n, with the decision taken on day t:

1) Structural forecast (from the cancellation model):
     base   = guests already in house + sum (1 - p_i) over bookings "on the books"
     pickup = rho * free capacity      (bookings still to come)
   The Poisson-binomial distribution assumes independent bookings.

2) Stage-2 correction model ("booking curve", LightGBM):
   learns how the current book of bookings translates into final hotel occupancy
   under the current sales practice. Uncertainty: empirical cross-validation
   residuals (conformal approach) - realistic intervals.

Decision: how many X extra bookings to sell on top of current practice.
Each extra booking materialises with probability q.
   Profit(X) = v * E[min(R + Y, Cap)] - w * E[max(R + Y - Cap, 0)],  Y ~ Bin(X, q)
v = room value (ADR + ancillary spend), w = cost of relocating a guest.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats

import config as C
from src.occupancy import pickup_forecast, pickup_rate_forecast


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------
def poisson_binomial_pmf(probs: np.ndarray) -> np.ndarray:
    """Exact distribution of a sum of independent Bernoulli(probs) (dynamic programming)."""
    pmf = np.array([1.0])
    for p in np.asarray(probs, dtype=float):
        new = np.zeros(len(pmf) + 1)
        new[:-1] += pmf * (1 - p)
        new[1:] += pmf * p
        pmf = new
    return pmf


def poisson_pmf(mu: float) -> np.ndarray:
    if mu <= 0:
        return np.array([1.0])
    kmax = int(mu + 10 * np.sqrt(mu) + 10)
    pmf = stats.poisson.pmf(np.arange(kmax + 1), mu)
    return pmf / pmf.sum()


def structural_distribution(in_house: int, show_probs: np.ndarray, mu_pickup: float) -> np.ndarray:
    pmf = np.convolve(poisson_binomial_pmf(show_probs), poisson_pmf(mu_pickup))
    return np.concatenate([np.zeros(in_house), pmf])


def pmf_quantile(pmf: np.ndarray, q: float) -> int:
    return int(np.searchsorted(np.cumsum(pmf), q))


def walk_cost(adr: float) -> float:
    return C.WALK_COST_ADR_MULTIPLIER * adr + C.WALK_COST_FIXED_EUR


# ---------------------------------------------------------------------------
# Optimisation of a single night
# ---------------------------------------------------------------------------
def optimize_night(pmf_r: np.ndarray, capacity: int, value: float, w_cost: float,
                   q: float, x_max: int = C.MAX_EXTRA_BOOKINGS) -> dict:
    k = np.arange(len(pmf_r) + x_max + 1)
    occ_val = value * np.minimum(k, capacity) - w_cost * np.maximum(k - capacity, 0)
    walks = np.maximum(k - capacity, 0)
    profits, p_walk, e_walk = [], [], []
    binom = np.array([1.0])
    for x in range(x_max + 1):
        if x > 0:
            b = np.zeros(len(binom) + 1)
            b[:-1] += binom * (1 - q)
            b[1:] += binom * q
            binom = b
        pmf_t = np.convolve(pmf_r, binom)
        n = len(pmf_t)
        profits.append(pmf_t @ occ_val[:n])
        e_walk.append(pmf_t @ walks[:n])
        p_walk.append(pmf_t[capacity + 1:].sum())
    profits = np.array(profits)
    x_best = int(np.argmax(profits))
    support = np.arange(len(pmf_r))
    return {
        "x_opt": x_best,
        "exp_gain": float(profits[x_best] - profits[0]),
        "p_walk_opt": float(p_walk[x_best]),
        "exp_walks_opt": float(e_walk[x_best]),
        "exp_occupancy": float(support @ pmf_r),
        "occ_p10": pmf_quantile(pmf_r, 0.10),
        "occ_p50": pmf_quantile(pmf_r, 0.50),
        "occ_p90": pmf_quantile(pmf_r, 0.90),
        "exp_empty_rooms": float(np.maximum(capacity - support, 0) @ pmf_r),
        "p_full": float(pmf_r[capacity:].sum()),
    }


# ---------------------------------------------------------------------------
# State of the booking book
# ---------------------------------------------------------------------------
def book_state(df: pd.DataFrame, model, hotel: str, nights: pd.DatetimeIndex,
               as_of: pd.Timestamp | None = None, horizon: int | None = None) -> pd.DataFrame:
    """(booking, night) pairs with their state at decision time: in_house / on_books + p.

    Backtest mode: horizon=h    -> decision for each night n on day t = n - h
    Forward mode:  as_of=date   -> decision for all nights on day t = as_of
    """
    g = df[df["hotel"] == hotel]
    g = g[(g["arrival"] <= nights.max()) & (g["departure"] > nights.min())]
    pairs = g.loc[g.index.repeat(g["nights"])].copy()
    pairs["k"] = pairs.groupby(level=0).cumcount()
    pairs["night"] = pairs["arrival"] + pd.to_timedelta(pairs["k"], unit="D")
    pairs = pairs[pairs["night"].isin(nights)]

    t = (pd.Series(as_of, index=pairs.index) if as_of is not None
         else pairs["night"] - pd.Timedelta(days=horizon))
    dta = (pairs["arrival"] - t).dt.days                  # days to arrival at time t

    arrived = dta < 0
    in_house = arrived & (pairs["reservation_status"] == "Check-Out")
    active = (~arrived) & (pairs["lead_time"] >= dta) & (
        (pairs["is_cancel"] == 0) | (pairs["cancel_days_before"] <= dta))

    pairs["state"] = np.where(in_house, "in_house", np.where(active, "on_books", "other"))
    pairs["dta"] = dta
    pairs = pairs[pairs["state"] != "other"].copy()
    pairs["p_no_arrival"] = 0.0
    m = pairs["state"] == "on_books"
    if m.any():
        pairs.loc[m, "p_no_arrival"] = model.predict(pairs[m], days_to_arrival=pairs.loc[m, "dta"])
    return pairs


def night_features(df, model, hotel, nights, capacity, pick, occ_s,
                   horizon=None, as_of=None) -> pd.DataFrame:
    """Booking-book aggregates per night (stage-2 model features)."""
    pairs = book_state(df, model, hotel, nights, as_of=as_of, horizon=horizon)
    pairs["s"] = 1 - pairs["p_no_arrival"]
    ob = pairs["state"] == "on_books"
    g = pairs.assign(
        ih=(~ob).astype(int), ob=ob.astype(int),
        e_ob=np.where(ob, pairs["s"], 0.0),
        v_ob=np.where(ob, pairs["s"] * (1 - pairs["s"]), 0.0),
        rev=pairs["adr"],
    ).groupby("night").agg(in_house=("ih", "sum"), on_books=("ob", "sum"),
                           e_ob=("e_ob", "sum"), v_ob=("v_ob", "sum"),
                           adr=("rev", "mean"))
    f = g.reindex(nights).fillna({"in_house": 0, "on_books": 0, "e_ob": 0.0, "v_ob": 0.0})
    f["adr"] = f["adr"].fillna(f["adr"].median() if f["adr"].notna().any() else 100.0)
    f.index.name = "night"
    f = f.reset_index()
    f["hotel"] = hotel
    f["capacity"] = capacity
    f["horizon"] = horizon if horizon is not None else (f["night"] - as_of).dt.days
    known = occ_s if as_of is None else occ_s[occ_s.index < as_of]
    f["base"] = f["in_house"] + f["e_ob"]
    f["rho"] = [pickup_rate_forecast(pick, known, capacity, n, int(h), as_of=as_of)
                for n, h in zip(f["night"], f["horizon"])]
    f["pickup_hist"] = [pickup_forecast(pick, n, int(h), as_of=as_of)
                        for n, h in zip(f["night"], f["horizon"])]
    f["pickup_struct"] = f["rho"] * np.maximum(capacity - f["base"], 0)
    f["struct_forecast"] = np.minimum(f["base"] + f["pickup_struct"], capacity)
    f["gap"] = capacity - f["base"]
    f["dow"] = f["night"].dt.dayofweek
    f["realized"] = occ_s.reindex(f["night"]).to_numpy()   # unknown in production
    return f


# ---------------------------------------------------------------------------
# Stage-2 model: final occupancy from the booking book
# ---------------------------------------------------------------------------
STAGE2_FEATURES = ["horizon", "base", "in_house", "on_books", "e_ob", "v_ob", "rho",
                   "pickup_hist", "pickup_struct", "gap", "dow", "capacity", "is_city"]
HORIZON_BUCKETS = [(0, 3), (4, 7), (8, 14), (15, 30), (31, 60), (61, 10_000)]


def horizon_bucket(h: int) -> str:
    for lo, hi in HORIZON_BUCKETS:
        if lo <= h <= hi:
            return f"D{lo}-{hi}" if hi < 10_000 else f"D{lo}+"
    return "D61+"


class OccupancyForecaster:
    params = dict(objective="l2", learning_rate=0.03, num_leaves=15,
                  min_child_samples=40, lambda_l2=5.0, verbose=-1, seed=C.RANDOM_STATE)
    rounds = 400

    @staticmethod
    def _X(f):
        X = f.copy()
        X["is_city"] = (X["hotel"] == "City Hotel").astype(int)
        return X[STAGE2_FEATURES]

    def fit(self, f: pd.DataFrame, folds: int = 5):
        f = f.dropna(subset=["realized"]).copy()
        y = f["realized"] - f["base"]
        # cross-validation in blocks of weeks -> out-of-sample residuals
        week = f["night"].dt.to_period("W").astype(str)
        uw = np.array(sorted(week.unique()))
        fold_of = {w: i % folds for i, w in enumerate(uw)}
        fid = week.map(fold_of).to_numpy()
        oof = np.zeros(len(f))
        for k in range(folds):
            tr, va = fid != k, fid == k
            b = lgb.train(self.params, lgb.Dataset(self._X(f[tr]), y[tr]), self.rounds)
            oof[va] = b.predict(self._X(f[va]))
        f["pred"] = np.minimum(f["base"] + oof, f["capacity"])
        f["resid"] = f["realized"] - f["pred"]
        f["bucket"] = f["horizon"].map(horizon_bucket)
        self.residuals_ = {k: g["resid"].to_numpy() for k, g in f.groupby(["hotel", "bucket"])}
        self.cv_table_ = (f.groupby(["hotel", "bucket"])
                           .agg(n=("resid", "size"), bias=("resid", "mean"),
                                mae=("resid", lambda r: r.abs().mean()),
                                p10=("resid", lambda r: r.quantile(0.1)),
                                p90=("resid", lambda r: r.quantile(0.9))))
        self.booster = lgb.train(self.params, lgb.Dataset(self._X(f), y), self.rounds)
        return self

    def predict_mean(self, f: pd.DataFrame) -> np.ndarray:
        return np.minimum(f["base"].to_numpy() + self.booster.predict(self._X(f)),
                          f["capacity"].to_numpy())

    def pmf(self, pred: float, hotel: str, h: int, capacity: int) -> np.ndarray:
        r = self.residuals_.get((hotel, horizon_bucket(h)))
        if r is None:
            r = np.concatenate([v for (ho, _), v in self.residuals_.items() if ho == hotel])
        vals = np.clip(np.round(pred + r), 0, capacity).astype(int)
        pmf = np.bincount(vals, minlength=capacity + 1).astype(float)
        return pmf / pmf.sum()


def add_decisions(f: pd.DataFrame, forecaster: OccupancyForecaster, q_curves: dict) -> pd.DataFrame:
    f = f.copy()
    f["forecast"] = forecaster.predict_mean(f)
    out = []
    for r in f.itertuples(index=False):
        h = int(r.horizon)
        q = float(q_curves[r.hotel][min(h, len(q_curves[r.hotel]) - 1)])
        value = r.adr + C.ANCILLARY_PER_ROOM_NIGHT.get(r.hotel, 0.0)
        pmf = forecaster.pmf(r.forecast, r.hotel, h, int(r.capacity))
        res = optimize_night(pmf, int(r.capacity), value, walk_cost(r.adr), q)
        res.update({"room_value": value, "walk_cost": walk_cost(r.adr), "q_extra_show": q})
        out.append(res)
    return pd.concat([f.reset_index(drop=True), pd.DataFrame(out)], axis=1)


def build_night_features(df, model, occ, capacity, picks, nights, horizons=None, as_of=None):
    parts = []
    for hotel in sorted(capacity):
        o = occ[occ["hotel"] == hotel].set_index("night")["occupied"]
        if as_of is not None:
            parts.append(night_features(df, model, hotel, nights, capacity[hotel],
                                        picks[hotel], o, as_of=as_of))
        else:
            for h in horizons:
                parts.append(night_features(df, model, hotel, nights, capacity[hotel],
                                            picks[hotel], o, horizon=h))
    return pd.concat(parts, ignore_index=True)


TRAIN_HORIZONS = [0, 1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60]


def fit_forecaster(df, model, occ, capacity, picks,
                   start="2016-07-15", end=C.VALID_END):
    """The stage-2 model is trained on nights of the validation period - there the
    cancellation model produces out-of-sample predictions (on the training period
    it would be overfitted)."""
    nights = pd.date_range(start, end, freq="D")
    f = build_night_features(df, model, occ, capacity, picks, nights, horizons=TRAIN_HORIZONS)
    return OccupancyForecaster().fit(f), f


# ---------------------------------------------------------------------------
# Backtest of overbooking policies on the test period
# ---------------------------------------------------------------------------
POLICY_NAMES = {
    "none": "No extra sales (current practice)",
    "static": f"Static overbooking {int(C.STATIC_OVERBOOKING_PCT * 100)}%",
    "model": "ML model (daily recommendation)",
    "oracle": "Oracle (upper bound)",
}


def backtest(df, model, occ, capacity, picks, q_curves, forecaster,
             start="2017-01-01", end=None, horizon=C.DECISION_HORIZON_DAYS,
             seed=C.RANDOM_STATE):
    end = end or C.NIGHTS_END
    rng = np.random.default_rng(seed)
    nights = pd.date_range(start, end, freq="D")
    f = build_night_features(df, model, occ, capacity, picks, nights, horizons=[horizon])
    dec = add_decisions(f, forecaster, q_curves)
    dec["realized"] = dec["realized"].fillna(0).astype(int)

    # --- [SYNTHETIC] turned-away demand on sold-out nights -------------------
    sold_out = dec["realized"] >= C.SOLD_OUT_THRESHOLD * dec["capacity"]
    dec["sold_out_night"] = sold_out
    dec["extra_demand"] = np.where(sold_out, rng.poisson(C.EXTRA_DEMAND_LAMBDA * dec["capacity"]), 0)
    U = rng.random((len(dec), C.MAX_EXTRA_BOOKINGS + 1))   # common random numbers

    def realize(x):
        accepted = np.minimum(x, dec["extra_demand"].to_numpy())
        q = dec["q_extra_show"].to_numpy()
        shows = np.array([(U[i, :a] < q[i]).sum() for i, a in enumerate(accepted)])
        total = dec["realized"].to_numpy() + shows
        cap = dec["capacity"].to_numpy()
        occupied = np.minimum(total, cap)
        walks = np.maximum(total - cap, 0)
        recovered = occupied - dec["realized"].to_numpy()
        gain = recovered * dec["room_value"].to_numpy() - walks * dec["walk_cost"].to_numpy()
        return recovered, walks, gain

    best = np.full(len(dec), -np.inf)
    best_x = np.zeros(len(dec), dtype=int)
    for x in range(C.MAX_EXTRA_BOOKINGS + 1):
        gain = realize(np.full(len(dec), x))[2]
        better = gain > best
        best[better], best_x[better] = gain[better], x

    policies = {"none": np.zeros(len(dec), dtype=int),
                "static": np.round(C.STATIC_OVERBOOKING_PCT * dec["capacity"]).astype(int).to_numpy(),
                "model": dec["x_opt"].to_numpy(),
                "oracle": best_x}
    for key, x in policies.items():
        rec, walks, gain = realize(x)
        dec[f"{key}_x"], dec[f"{key}_recovered"] = x, rec
        dec[f"{key}_walks"], dec[f"{key}_gain"] = walks, gain

    summary = []
    for key, name in POLICY_NAMES.items():
        summary.append({
            "Policy": name,
            "Recovered room-nights": int(dec[f"{key}_recovered"].sum()),
            "Guests walked": int(dec[f"{key}_walks"].sum()),
            "Nights with walks": int((dec[f"{key}_walks"] > 0).sum()),
            "Revenue from recovered rooms [EUR]": float((dec[f"{key}_recovered"] * dec["room_value"]).sum()),
            "Walk cost [EUR]": float((dec[f"{key}_walks"] * dec["walk_cost"]).sum()),
            "Net result [EUR]": float(dec[f"{key}_gain"].sum()),
        })
    summary = pd.DataFrame(summary).set_index("Policy")
    top = summary.loc[POLICY_NAMES["oracle"], "Net result [EUR]"]
    summary["% of potential"] = summary["Net result [EUR]"] / top if top > 0 else np.nan
    return dec, summary


def forecast_accuracy(df, model, occ, capacity, picks, forecaster,
                      start="2017-01-05", end="2017-06-30", step_days=14, days=60):
    """Occupancy forecast accuracy over the test period for different horizons:
    structural forecast vs stage-2 model vs naive (occupancy one year earlier)."""
    recs = []
    occ_i = occ.set_index(["hotel", "night"])["occupied"]
    for as_of in pd.date_range(start, end, freq=f"{step_days}D"):
        nights = pd.date_range(as_of, as_of + pd.Timedelta(days=days - 1))
        f = build_night_features(df, model, occ, capacity, picks, nights, as_of=as_of)
        f["forecast"] = forecaster.predict_mean(f)
        ly = [occ_i.get((h, n - pd.Timedelta(days=364)), np.nan) for h, n in zip(f["hotel"], f["night"])]
        f["naive_last_year"] = ly
        recs.append(f)
    r = pd.concat(recs, ignore_index=True).dropna(subset=["realized"])
    r["bucket"] = r["horizon"].map(horizon_bucket)
    order = [horizon_bucket(lo) for lo, _ in HORIZON_BUCKETS]
    acc = (r.groupby(["hotel", "bucket"])
            .apply(lambda g: pd.Series({
                "MAE: naive (last year)": (g["naive_last_year"] - g["realized"]).abs().mean(),
                "MAE: structural": (g["struct_forecast"] - g["realized"]).abs().mean(),
                "MAE: stage-2 model": (g["forecast"] - g["realized"]).abs().mean(),
                "Nights": len(g),
            })))
    acc = acc.reset_index()
    acc["bucket"] = pd.Categorical(acc["bucket"], [b for b in order if b in set(acc["bucket"])], ordered=True)
    acc = acc.sort_values(["hotel", "bucket"]).set_index(["hotel", "bucket"])
    acc.index.names = ["Hotel", "Horizon"]
    return acc, r


# ---------------------------------------------------------------------------
# Recommendations for the coming days (production mode)
# ---------------------------------------------------------------------------
def forward_plan(df, model, occ, capacity, picks, q_curves, forecaster,
                 as_of, days=C.FORWARD_DAYS):
    as_of = pd.Timestamp(as_of)
    nights = pd.date_range(as_of, as_of + pd.Timedelta(days=days - 1), freq="D")
    f = build_night_features(df, model, occ, capacity, picks, nights, as_of=as_of)
    plan = add_decisions(f, forecaster, q_curves)

    def status(r):
        if r["exp_empty_rooms"] >= 0.08 * r["capacity"]:
            return "Low occupancy - sales push / promotion"
        if r["x_opt"] > 0:
            return f"Allow overbooking: +{int(r['x_opt'])} bookings"
        return "No overbooking - risk of walks"
    plan["recommendation"] = plan.apply(status, axis=1)
    return plan
