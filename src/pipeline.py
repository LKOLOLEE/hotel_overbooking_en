"""Orchestration of the whole project: data -> model -> overbooking -> actions -> report."""
from __future__ import annotations

import json
import time

import joblib

import matplotlib.pyplot as plt
import pandas as pd

import config as C
from src import actions, business_case, eda, explain, monitoring, outreach, policy
from src import overbooking as ob
from src import overbooking_plots as obp
from src.data import load_clean
from src.features import build_snapshots
from src.modeling import train_and_evaluate
from src.occupancy import (estimate_capacity, last_minute_show_rate,
                           nightly_occupancy, pickup_matrix)
from src.plotting import save


def assumptions_table() -> pd.DataFrame:
    rows = [
        ("Hotel capacity", "max. realised occupancy" if not C.CAPACITY_OVERRIDE else str(C.CAPACITY_OVERRIDE),
         "Estimated from data (the dataset has no room count)"),
        ("Cost of walking a guest", f"{C.WALK_COST_ADR_MULTIPLIER} x ADR + EUR {C.WALK_COST_FIXED_EUR:.0f}", "SYNTHETIC"),
        ("Ancillary spend per room-night", str(C.ANCILLARY_PER_ROOM_NIGHT), "SYNTHETIC"),
        ("Sold-out night (turned-away demand exists)", f">= {C.SOLD_OUT_THRESHOLD:.0%} of capacity", "SYNTHETIC"),
        ("Turned-away demand on a sold-out night", f"Poisson({C.EXTRA_DEMAND_LAMBDA:.0%} of capacity)", "SYNTHETIC"),
        ("Overbooking decision time", f"D-{C.DECISION_HORIZON_DAYS}", "Parameter"),
        ("Benchmark policy", f"static overbooking {C.STATIC_OVERBOOKING_PCT:.0%}", "Parameter"),
        ("Max. effect of guest contact", f"{C.OUTREACH_MAX_EFFECT:.0%} fewer cancellations", "SYNTHETIC"),
        ("Cost per contact", f"EUR {C.OUTREACH_COST_PER_CONTACT:.0f}", "SYNTHETIC"),
        ("Prepayment discount / take-up rate",
         f"{C.OUTREACH_DISCOUNT_PCT:.0%} / {C.OUTREACH_DISCOUNT_TAKE_UP:.0%}", "SYNTHETIC"),
        ("Chance of reselling a released room", f"{C.RESALE_PROBABILITY:.0%}", "SYNTHETIC"),
        ("Effect of a prepayment requirement", f"cancellations -{C.POLICY_CANCEL_REDUCTION:.0%}, bookings -{C.POLICY_BOOKING_LOSS:.0%}", "SYNTHETIC"),
        ("Resale of an early-released room / replacement of a lost booking",
         f"{C.POLICY_EARLY_RESALE:.0%} / {C.POLICY_REPLACEMENT:.0%}", "SYNTHETIC"),
        ("Project / running cost", f"EUR {C.PROJECT_COST_EUR:,.0f} / EUR {C.RUN_COST_EUR_PER_YEAR:,.0f} per year", "SYNTHETIC"),
        ("Time-based split", f"training to {C.TRAIN_END}, validation to {C.VALID_END}, test 2017-01..08", "Methodology"),
    ]
    return pd.DataFrame(rows, columns=["Assumption", "Value", "Source"]).set_index("Assumption")


def run_all(as_of: str = C.DEMO_AS_OF, save_outputs: bool = True, verbose: bool = True) -> dict:
    T, F = {}, {}
    t0 = time.time()

    def log(msg):
        if verbose:
            print(f"[{time.time() - t0:6.1f}s] {msg}")

    for d in (C.FIGURES, C.TABLES, C.MODELS, C.REPORTS):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Data ---------------------------------------------------------------
    log("1/9 Loading and cleaning data")
    df = load_clean()
    years_all = (df["arrival"].max() - df["arrival"].min()).days / 365
    T["eda_summary"] = eda.summary_table(df)
    F["eda_trend"] = eda.fig_cancel_trend(df)
    F["eda_lead"] = eda.fig_cancel_by_lead(df)
    F["eda_segment"] = eda.fig_cancel_by_segment(df)
    F["eda_timing"] = eda.fig_cancel_timing(df)

    occ = nightly_occupancy(df)
    capacity = estimate_capacity(occ)
    F["eda_occupancy"] = eda.fig_occupancy(occ, capacity)

    # 2. Snapshots and model -----------------------------------------------------
    log("2/9 Leakage-free snapshots + model training")
    snap = build_snapshots(df)
    model, te, comp, by_h = train_and_evaluate(snap, verbose=verbose)
    model.save()
    T["model_comparison"], T["model_by_horizon"] = comp, by_h
    F["model_roc"] = explain.fig_roc(te)
    F["model_calibration"] = explain.fig_calibration(te)
    F["model_auc_horizon"] = explain.fig_auc_by_horizon(by_h)

    # 3. Explainability -------------------------------------------------------
    log("3/9 Explainability (SHAP)")
    _, Xs, contrib = explain.shap_sample(model, te)
    T["feature_importance"] = explain.importance_table(contrib)
    F["shap_importance"] = explain.fig_importance(contrib)
    F["shap_dependence"] = explain.fig_dependence(Xs, contrib)
    F["shap_categories"] = explain.fig_category_effects(Xs, contrib)

    # 4. Occupancy forecast ----------------------------------------------------
    log("4/9 Occupancy forecast (stage-2 model)")
    picks = pickup_matrix(df)
    q_curves = last_minute_show_rate(df)
    forecaster, _ = ob.fit_forecaster(df, model, occ, capacity, picks)
    acc, _ = ob.forecast_accuracy(df, model, occ, capacity, picks, forecaster)
    T["forecast_accuracy"] = acc
    T["forecast_cv"] = forecaster.cv_table_
    F["forecast_accuracy"] = obp.fig_forecast_accuracy(acc)

    # 5. Overbooking backtest --------------------------------------------------
    log("5/9 Backtest of overbooking policies (2017)")
    dec, bt = ob.backtest(df, model, occ, capacity, picks, q_curves, forecaster)
    T["backtest_summary"], T["backtest_nights"] = bt, dec
    F["ob_example"] = obp.fig_example_night(dec, forecaster)
    F["ob_cumulative"] = obp.fig_backtest_cumulative(dec)
    F["ob_bars"] = obp.fig_backtest_bars(bt)

    # 6. Preventive actions (synthetic A/B) -------------------------------
    log("6/9 A/B experiment on preventive actions [synthetic]")
    exp = outreach.simulate_experiment(te)
    ov, by_dec = outreach.analyze(exp)
    test_years = exp["arrival"].dt.normalize().nunique() / 365
    outreach_annual = outreach.annual_value(exp, ov, test_years)
    T["outreach_overall"], T["outreach_deciles"] = ov, by_dec
    F["outreach_uplift"] = outreach.fig_uplift(by_dec)

    # 7. Cancellation policies ------------------------------------------------------
    log("7/9 Segment and policy analysis")
    seg = policy.segment_table(df)
    wi = policy.whatif_prepayment(seg, years_all)
    T["policy_whatif"], T["channels"] = wi, policy.channel_table(df, years_all)
    policy_annual = float(wi.loc[wi["Revenue change per year [EUR]"] > 0, "Revenue change per year [EUR]"].sum())
    F["policy_heat"] = policy.fig_channel_heat(df)
    F["policy_whatif"] = policy.fig_whatif(wi)

    # 8. Production mode: plan and risk list -----------------------------------
    log(f"8/9 Overbooking plan and risk list as of {as_of}")
    plan = ob.forward_plan(df, model, occ, capacity, picks, q_curves, forecaster, as_of)
    thr = ov.get(outreach.THRESHOLD_KEY, 0.25)
    risk, risk_sum = actions.risk_list(df, model, as_of, contact_threshold=thr)
    T["forward_plan"], T["risk_list"], T["risk_summary"] = plan, risk, risk_sum
    joblib.dump(forecaster, C.MODELS / "occupancy_forecaster.joblib")
    (C.MODELS / "settings.json").write_text(json.dumps({
        "contact_threshold": None if pd.isna(thr) else float(thr),
        "capacity": {k: int(v) for k, v in capacity.items()},
    }, indent=1), encoding="utf-8")
    F["forward_plan"] = obp.fig_forward_plan(plan, as_of)

    # 9. Monitoring and business case ------------------------------------------------
    log("9/9 Monitoring and business case")
    T["drift"] = monitoring.drift_table(model, snap)
    T["monthly"] = monitoring.monthly_performance(te)
    F["drift"] = monitoring.fig_drift(T["drift"])
    F["monthly"] = monitoring.fig_monthly(T["monthly"])
    items, kpi = business_case.build(bt, dec["night"].nunique(), outreach_annual, policy_annual)
    T["business_items"], T["kpi"] = items, kpi
    F["business"] = business_case.fig_waterfall(items)
    T["assumptions"] = assumptions_table()
    T["meta"] = {"as_of": as_of, "capacity": capacity, "n_bookings": len(df),
                 "n_snapshots": len(snap), "best_iteration": model.booster.best_iteration}

    if save_outputs:
        for name, fig in F.items():
            save(fig, name)
        for name in ["eda_summary", "model_comparison", "model_by_horizon", "feature_importance",
                     "forecast_accuracy", "backtest_summary", "backtest_nights", "outreach_deciles",
                     "policy_whatif", "channels", "forward_plan", "risk_list", "drift", "monthly",
                     "business_items", "assumptions"]:
            T[name].to_csv(C.TABLES / f"{name}.csv", encoding="utf-8-sig")
    for fig in F.values():
        plt.close(fig)
    log("Done.")
    return {"tables": T, "figures": F, "model": model, "forecaster": forecaster,
            "test_snapshots": te, "experiment": exp, "snapshots": snap,
            "df": df, "occ": occ, "capacity": capacity, "picks": picks, "q_curves": q_curves}
