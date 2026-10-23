"""Model evaluation (charts) and explainability (SHAP values from LightGBM)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

import config as C
from src.data import CATEGORICAL
from src.features import FEATURE_LABELS
from src.plotting import COL, pct_axis, titles

MODEL_STYLE = {
    "p_segment": ("Segment heuristic", COL["neutral"]),
    "p_logistic": ("Logistic regression", COL["s2"]),
    "p_lgbm": ("LightGBM + calibration", COL["s1"]),
}


# ---------------------------------------------------------------------------
# Model quality
# ---------------------------------------------------------------------------
def fig_roc(te):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for col, (name, color) in MODEL_STYLE.items():
        fpr, tpr, _ = roc_curve(te["no_arrival"], te[col])
        auc = roc_auc_score(te["no_arrival"], te[col])
        ax.plot(fpr, tpr, color=color, label=f"{name} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], color=COL["grid"], lw=1, ls="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("Share of no-arrivals detected")
    ax.legend(loc="lower right")
    titles(ax, "ROC curve - 2017 test period", "The closer to the top-left corner, the better")
    return fig


def fig_calibration(te, bins=10):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    top = 0.0
    for col in ["p_segment", "p_lgbm"]:
        name, color = MODEL_STYLE[col]
        q = pd.qcut(te[col].rank(method="first"), bins, labels=False)
        g = te.groupby(q).agg(p=(col, "mean"), y=("no_arrival", "mean"))
        ax.plot(g["p"], g["y"], marker="o", ms=5, color=color, label=name)
        top = max(top, g["p"].max(), g["y"].max())
    ax.plot([0, 1], [0, 1], color=COL["grid"], lw=1, ls="--")
    lim = min(1.0, top * 1.15)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    pct_axis(ax)
    pct_axis(ax, "x")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Actual no-arrival rate")
    ax.legend(loc="upper left")
    titles(ax, "Calibration: predicted = actual",
           "Points on the diagonal mean trustworthy probabilities")
    return fig


def fig_auc_by_horizon(by_h):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(by_h))
    ax.plot(x, by_h["ROC AUC (heuristic)"], marker="o", ms=5, color=COL["neutral"],
            label="Segment heuristic")
    ax.plot(x, by_h["ROC AUC (LightGBM)"], marker="o", ms=5, color=COL["s1"],
            label="LightGBM + calibration")
    ax.set_xticks(x, [f"D-{h}" for h in by_h.index])
    ax.set_xlabel("Prediction time (days before arrival)")
    ax.set_ylabel("ROC AUC")
    ax.legend(loc="lower left")
    titles(ax, "The model wins at every horizon",
           "ROC AUC on the test period by prediction time")
    return fig


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------
def shap_sample(model, te, n=20000, seed=C.RANDOM_STATE):
    s = te.sample(min(n, len(te)), random_state=seed)
    contrib = model.contributions(s)
    X = model.features(s)
    return s, X, contrib


def importance_table(contrib: pd.DataFrame) -> pd.DataFrame:
    imp = contrib.abs().mean().sort_values(ascending=False)
    out = pd.DataFrame({"Feature": [FEATURE_LABELS.get(c, c) for c in imp.index],
                        "Mean |SHAP|": imp.values}, index=imp.index)
    out["Share"] = out["Mean |SHAP|"] / out["Mean |SHAP|"].sum()
    return out


def fig_importance(contrib, top=15):
    imp = importance_table(contrib).head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(imp["Feature"], imp["Share"], color=COL["s1"], height=0.65)
    pct_axis(ax, "x")
    ax.grid(axis="y", visible=False)
    titles(ax, "What drives the risk of no-arrival the most",
           "Share of mean absolute impact (SHAP), top 15 features")
    return fig


def fig_dependence(X, contrib):
    feats = ["lead_time", "days_to_arrival", "previous_cancellations"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, f in zip(axes, feats):
        ax.scatter(X[f], contrib[f], s=4, alpha=0.25, color=COL["s1"], edgecolors="none")
        ax.axhline(0, color=COL["neutral"], lw=1)
        ax.set_xlabel(FEATURE_LABELS[f])
        if f in ("lead_time", "days_to_arrival"):
            ax.set_xlim(0, np.percentile(X[f], 99))
        if f == "previous_cancellations":
            ax.set_xlim(-0.5, 5)
    axes[0].set_ylabel("Impact on risk (SHAP, log-odds)")
    fig.suptitle("How numeric features change risk (above zero = higher risk)",
                 x=0.06, ha="left", fontsize=12, fontweight="bold", color=COL["text"])
    fig.tight_layout()
    return fig


def fig_category_effects(X, contrib):
    feats = ["market_segment", "deposit_type", "customer_type"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, f in zip(axes, feats):
        g = (pd.DataFrame({"v": X[f].astype(str), "s": contrib[f]})
               .groupby("v")["s"].agg(["mean", "size"]))
        g = g[g["size"] >= 50].sort_values("mean")
        colors = [COL["critical"] if v > 0 else COL["s1"] for v in g["mean"]]
        ax.barh(g.index, g["mean"], color=colors, height=0.6)
        ax.axvline(0, color=COL["neutral"], lw=1)
        ax.grid(axis="y", visible=False)
        ax.set_title(FEATURE_LABELS[f], fontsize=10, pad=4)
    axes[0].set_xlabel("Mean impact (SHAP)")
    fig.suptitle("Category effects: red increases risk, blue decreases it",
                 x=0.06, ha="left", fontsize=12, fontweight="bold", color=COL["text"])
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Risk reasons for a single booking (list for the front office)
# ---------------------------------------------------------------------------
def _fmt(feature, value):
    label = FEATURE_LABELS.get(feature, feature)
    if feature in CATEGORICAL:
        return f"{label}: {value}"
    if feature in ("has_company", "is_repeated_guest"):
        return f"{label}: {'yes' if value >= 1 else 'no'}"
    if feature == "arrival_dow":
        return f"{label}: {['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][int(value)]}"
    return f"{label}: {value:g}"


def risk_reasons(model, rows, days_to_arrival=None, top=3) -> list[str]:
    contrib = model.contributions(rows, days_to_arrival)
    X = model.features(rows, days_to_arrival)
    out = []
    for i in range(len(rows)):
        c = contrib.iloc[i]
        pos = c[c > 0].sort_values(ascending=False).head(top)
        out.append("; ".join(_fmt(f, X.iloc[i][f]) for f in pos.index))
    return out
