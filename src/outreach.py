"""
Preventive actions: A/B experiment contacting guests at D-14.  [SYNTHETIC]

The public dataset contains no actions taken towards guests, so the effect of
contact (an e-mail asking for confirmation + a discounted prepayment offer /
a date change instead of a cancellation) is SIMULATED on real bookings:

  * a random 50% of bookings active at D-14 go to the treatment group,
  * in the treatment group an actual cancellation is "saved" with
    probability r(p) = MAX_EFFECT * 4 * p * (1 - p)
    (largest effect for "undecided" guests, p ~ 0.5; for bookings that are almost
    certain or almost lost anyway, contact changes little).

The analysis shows how to evaluate the effect in a real deployment (difference
in rates, confidence intervals, effect by risk decile) and how to decide whom it
pays to contact. After go-live the synthetic effect is replaced by a real test.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import config as C
from src.plotting import COL, eur_axis, pct_axis, titles

THRESHOLD_KEY = "Recommended risk threshold for contact"


def simulate_experiment(te: pd.DataFrame, seed=C.RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 7)
    e = te[te["days_to_arrival"] == C.OUTREACH_HORIZON_DAYS].copy()
    e["treated"] = (rng.random(len(e)) < C.OUTREACH_TREATMENT_SHARE).astype(int)
    p = e["p_lgbm"].to_numpy()
    effect = C.OUTREACH_MAX_EFFECT * 4 * p * (1 - p)
    effect = np.where(e["deposit_type"] == "Non Refund", effect * 0.2, effect)
    e["true_effect"] = effect                                   # for checking only
    saved = (e["treated"] == 1) & (e["no_arrival"] == 1) & (rng.random(len(e)) < effect)
    e["outcome"] = np.where(saved, 0, e["no_arrival"])
    e["risk_decile"] = pd.qcut(e["p_lgbm"].rank(method="first"), 10, labels=False) + 1
    return e


def _diff_ci(yc, yt):
    pc, pt = yc.mean(), yt.mean()
    se = np.sqrt(pc * (1 - pc) / len(yc) + pt * (1 - pt) / len(yt))
    d = pc - pt
    return d, d - 1.96 * se, d + 1.96 * se


def _uplift_model(e: pd.DataFrame) -> np.ndarray:
    """Smooth uplift model as a function of risk:
    logit P(no-arrival) = b0 + b1*lp + t*(c0 + c1*lp),  lp = logit(p).
    Reduces the noise of individual deciles when deciding whom to contact."""
    p = e["p_lgbm"].clip(1e-4, 1 - 1e-4).to_numpy()
    lp = np.log(p / (1 - p))
    t = e["treated"].to_numpy()

    def design(tt):
        return np.column_stack([lp, tt, tt * lp])
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(design(t), e["outcome"])
    return lr.predict_proba(design(np.zeros_like(t)))[:, 1] - lr.predict_proba(design(np.ones_like(t)))[:, 1]


def analyze(e: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    e["uplift_model"] = _uplift_model(e)
    e["net_value_model"] = (e["uplift_model"] * e["revenue"] * (1 - C.RESALE_PROBABILITY)
                            - C.OUTREACH_COST_PER_CONTACT
                            - C.OUTREACH_DISCOUNT_TAKE_UP * C.OUTREACH_DISCOUNT_PCT * e["revenue"]
                            * (1 - e["p_lgbm"]))
    d, lo, hi = _diff_ci(e.loc[e["treated"] == 0, "outcome"], e.loc[e["treated"] == 1, "outcome"])
    overall = {"Bookings in experiment": len(e),
               "No-arrival rate - control": e.loc[e["treated"] == 0, "outcome"].mean(),
               "No-arrival rate - contacted": e.loc[e["treated"] == 1, "outcome"].mean(),
               "Reduction (pp)": d, "95% CI lower": lo, "95% CI upper": hi}

    rows = []
    for dec, g in e.groupby("risk_decile"):
        c, t = g[g["treated"] == 0], g[g["treated"] == 1]
        up, ulo, uhi = _diff_ci(c["outcome"], t["outcome"])
        rev = g["revenue"].mean()
        up_m = g["uplift_model"].mean()
        value_saved = (g["uplift_model"] * g["revenue"]).mean() * (1 - C.RESALE_PROBABILITY)
        cost = C.OUTREACH_COST_PER_CONTACT + C.OUTREACH_DISCOUNT_TAKE_UP * C.OUTREACH_DISCOUNT_PCT * (
            g["revenue"] * (1 - g["p_lgbm"])).mean()
        rows.append({"Risk decile": dec,
                     "Mean predicted risk": g["p_lgbm"].mean(),
                     "Bookings": len(g),
                     "No-arrivals - control": c["outcome"].mean(),
                     "No-arrivals - contacted": t["outcome"].mean(),
                     "Uplift (pp)": up, "CI lower": ulo, "CI upper": uhi,
                     "Uplift - smoothed model": up_m,
                     "Avg. booking revenue [EUR]": rev,
                     "Value saved / contact [EUR]": value_saved,
                     "Cost / contact [EUR]": cost,
                     "Net result / contact [EUR]": value_saved - cost})
    by_dec = pd.DataFrame(rows).set_index("Risk decile")
    # threshold: the lowest risk from which (upwards) the average net value of contact is > 0
    s = e.sort_values("p_lgbm", ascending=False)
    cum = s["net_value_model"].cumsum().to_numpy()
    k = int(np.argmax(cum))
    thr = float(s["p_lgbm"].iloc[k]) if cum[k] > 0 else np.nan
    overall[THRESHOLD_KEY] = thr
    overall["Deciles to contact"] = ", ".join(
        str(i) for i in by_dec.index if not np.isnan(thr) and by_dec.loc[i, "Mean predicted risk"] >= thr)
    overall["Share of bookings to contact"] = float((e["p_lgbm"] >= thr).mean()) if not np.isnan(thr) else 0.0
    return overall, by_dec


def annual_value(e, overall, years) -> float:
    """Annual net value of contacting bookings above the risk threshold."""
    thr = overall[THRESHOLD_KEY]
    if np.isnan(thr):
        return 0.0
    return float(e.loc[e["p_lgbm"] >= thr, "net_value_model"].sum() / years)


def fig_uplift(by_dec):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x = by_dec.index.to_numpy()
    ax = axes[0]
    ax.bar(x, by_dec["Uplift (pp)"], color=COL["s1"], width=0.65)
    ax.errorbar(x, by_dec["Uplift (pp)"],
                yerr=[by_dec["Uplift (pp)"] - by_dec["CI lower"],
                      by_dec["CI upper"] - by_dec["Uplift (pp)"]],
                fmt="none", ecolor=COL["text2"], elinewidth=1, capsize=3)
    ax.plot(x, by_dec["Uplift - smoothed model"], marker="o", ms=5, color=COL["s2"],
            label="Smoothed model")
    ax.axhline(0, color=COL["neutral"], lw=1)
    ax.legend(loc="upper left")
    pct_axis(ax, decimals=0)
    ax.set_xticks(x)
    ax.set_xlabel("Predicted risk decile (10 = highest)")
    ax.grid(axis="x", visible=False)
    titles(ax, "Effect of contact by risk decile", "Reduction in the no-arrival rate with 95% CI")
    ax = axes[1]
    v = by_dec["Net result / contact [EUR]"]
    ax.bar(x, v, color=[COL["good"] if a > 0 else COL["critical"] for a in v], width=0.65)
    ax.axhline(0, color=COL["neutral"], lw=1)
    eur_axis(ax)
    ax.set_xticks(x)
    ax.set_xlabel("Predicted risk decile")
    ax.grid(axis="x", visible=False)
    titles(ax, "Whom it pays to contact", "Net result per contact [EUR]")
    fig.text(0.01, -0.03, "Experiment data is SYNTHETIC (simulated on real bookings).",
             fontsize=8, color=COL["text2"])
    fig.tight_layout()
    return fig
