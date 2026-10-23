"""
Inspection of the trained models: LightGBM (+ calibration), logistic regression,
segment heuristic and the stage-2 occupancy model.

How to run:
  * PyCharm / Spyder / VS Code: run the "# %%" cells in order (Shift+Enter / "Run Cell"),
    charts in the Plots / SciView pane, tables in the console.
  * Terminal / PyCharm "Run": python inspect_models.py  (charts are saved to outputs/inspect/)

If a model is missing (e.g. logistic regression from an earlier project version),
it is trained and saved automatically.
"""
# %% 0. Setup
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(ROOT))

import matplotlib
if not hasattr(sys, "ps1") and "ipykernel" not in sys.modules and "IPython" not in sys.modules:
    matplotlib.use("Agg")                       # run as a script: save to files
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd

import config as C
from src.data import CATEGORICAL, load_clean
from src.features import FEATURE_LABELS, FEATURES, build_snapshots, make_features
from src.modeling import (CancellationModel, SegmentRateModel, _lr_frame, metrics,
                          train_logistic)
from src.plotting import COL, titles

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 60)
OUT = C.OUTPUTS / "inspect"
OUT.mkdir(parents=True, exist_ok=True)


def show_fig(fig, name):
    fig.savefig(OUT / f"{name}.png")
    if matplotlib.get_backend().lower() == "agg":
        plt.close(fig)
    else:
        plt.show()


def header(txt):
    print("\n" + "=" * 100 + f"\n{txt}\n" + "=" * 100)


# %% 1. Load models and test data
if not (C.MODELS / "lgbm_model.txt").exists():
    sys.exit("No models found - run first: python run_pipeline.py")

model = CancellationModel.load()                       # LightGBM + calibrator
df = load_clean()
snap = build_snapshots(df)
tr = snap[snap["split"] == "train"]
va = snap[snap["split"] == "valid"]
te = snap[snap["split"] == "test"].copy()

lr_path, seg_path = C.MODELS / "logistic_regression.joblib", C.MODELS / "segment_heuristic.joblib"
if not lr_path.exists() or not seg_path.exists():
    print("No saved logistic regression / heuristic - training (approx. 10-20 s) ...")
    joblib.dump(train_logistic(make_features(tr), tr["no_arrival"]), lr_path)
    joblib.dump(SegmentRateModel().fit(pd.concat([tr, va])), seg_path)
lr = joblib.load(lr_path)
seg = joblib.load(seg_path)

te["p_lgbm"] = model.predict(te)
te["p_lgbm_raw"] = model.predict_raw(te)
te["p_logistic"] = lr.predict_proba(_lr_frame(make_features(te)))[:, 1]
te["p_segment"] = seg.predict(te)

header("MODEL FILES (outputs/models)")
for f in sorted(C.MODELS.iterdir()):
    print(f"  {f.name:35s} {f.stat().st_size / 1024:8.1f} KB")

header("COMPARISON ON THE 2017 TEST PERIOD")
print(pd.DataFrame({
    "Segment heuristic": metrics(te["no_arrival"], te["p_segment"]),
    "Logistic regression": metrics(te["no_arrival"], te["p_logistic"]),
    "LightGBM (raw)": metrics(te["no_arrival"], te["p_lgbm_raw"]),
    "LightGBM + calibration": metrics(te["no_arrival"], te["p_lgbm"]),
}).T.round(4))

# %% 2. LightGBM - parameters and structure
b = model.booster
header("LIGHTGBM - PARAMETERS AND STRUCTURE")
params = {k: v for k, v in b.params.items() if k in C.LGBM_PARAMS or k in ("num_iterations",)}
print("Training parameters:", {**C.LGBM_PARAMS, **params})
trees = b.trees_to_dataframe()
print(f"Number of trees (best_iteration): {b.num_trees()}")
print(f"Number of features: {b.num_feature()}  ->  {b.feature_name()}")
per_tree = trees.groupby("tree_index").agg(leaves=("node_depth", lambda d: (trees.loc[d.index, 'left_child'].isna()).sum()),
                                           depth=("node_depth", "max"))
print(f"Leaves per tree: mean {per_tree['leaves'].mean():.1f} (num_leaves={C.LGBM_PARAMS['num_leaves']}), "
      f"max. tree depth: {per_tree['depth'].max() - 1}")
print("Categories encoded in the model (from training):")
for c in CATEGORICAL:
    lv = model.categories[c]
    print(f"  {c:22s} ({len(lv):2d}): {', '.join(lv[:12])}{' ...' if len(lv) > 12 else ''}")

# %% 3. LightGBM - feature importance (gain vs split)
imp = pd.DataFrame({
    "Feature": [FEATURE_LABELS.get(f, f) for f in b.feature_name()],
    "Gain": b.feature_importance("gain"),
    "Split": b.feature_importance("split"),
}, index=b.feature_name())
imp["Gain %"] = imp["Gain"] / imp["Gain"].sum()
imp["Split %"] = imp["Split"] / imp["Split"].sum()
imp = imp.sort_values("Gain", ascending=False)
header("LIGHTGBM - FEATURE IMPORTANCE")
print(imp[["Feature", "Gain %", "Split %", "Split"]].round(3).to_string())

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
top = imp.head(15).iloc[::-1]
axes[0].barh(top["Feature"], top["Gain %"], color=COL["s1"], height=0.65)
axes[1].barh(top["Feature"], top["Split %"], color=COL["s2"], height=0.65)
titles(axes[0], "Gain", "Share of total loss reduction")
titles(axes[1], "Split", "Share of the number of splits in the trees")
for a in axes:
    a.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x * 100:g}%"))
    a.grid(axis="y", visible=False)
fig.tight_layout()
show_fig(fig, "lgbm_importance")

# %% 4. LightGBM - the first tree as text (no graphviz needed)
def print_tree(trees, tree_index=0, max_depth=3):
    t = trees[trees["tree_index"] == tree_index].set_index("node_index")

    def rec(node, depth, prefix):
        r = t.loc[node]
        if pd.isna(r["left_child"]):
            print(f"{prefix}leaf: log-odds {r['value']:+.3f}  (n={int(r['count'])})")
            return
        if depth >= max_depth:
            print(f"{prefix}... (subtree, n={int(r['count'])})")
            return
        thr, feat = r["threshold"], r["split_feature"]
        if r["decision_type"] == "==":                 # categorical split: codes -> names
            levels = model.categories[feat]
            names = [levels[int(k)] for k in str(thr).split("||") if int(k) < len(levels)]
            cond = f"{feat} ∈ {{{', '.join(names[:6])}{', ...' if len(names) > 6 else ''}}}"
        else:
            cond = f"{feat} {r['decision_type']} {0 if abs(float(thr)) < 1e-9 else float(thr):.4g}"
        print(f"{prefix}[{cond}]  gain={r['split_gain']:.0f}, n={int(r['count'])}")
        rec(r["left_child"], depth + 1, prefix + "   yes → ")
        rec(r["right_child"], depth + 1, prefix + "   no  → ")
    rec(f"{tree_index}-S0", 0, "")


header("LIGHTGBM - TREE 0 (first 3 levels)")
print_tree(trees, 0, max_depth=3)
print("\nFull structure of all trees: the 'trees' variable (DataFrame) or the file outputs/models/lgbm_model.txt")
trees.to_csv(OUT / "lgbm_trees.csv", index=False)
# Graphical version (requires: pip install graphviz + the Graphviz program on PATH):
# import lightgbm as lgb; lgb.plot_tree(b, tree_index=0, figsize=(30, 12), show_info=["split_gain", "leaf_count"]); plt.show()

# %% 5. Calibration: how the raw LightGBM score is turned into a probability
cal = model.calibrator
header("CALIBRATOR (isotonic regression)")
print(f"Number of steps in the step function: {len(cal.X_thresholds_)}")
grid = np.linspace(0, 1, 11)
print(pd.DataFrame({"Raw LightGBM score": grid, "After calibration": cal.predict(grid)}).round(3).to_string(index=False))

fig, ax = plt.subplots(figsize=(5.5, 5))
xs = np.linspace(0, 1, 400)
ax.plot(xs, cal.predict(xs), color=COL["s1"], label="Calibration function")
ax.plot([0, 1], [0, 1], color=COL["grid"], ls="--", lw=1, label="No correction")
ax.set_xlabel("Raw LightGBM probability")
ax.set_ylabel("Calibrated probability")
ax.legend(loc="upper left")
titles(ax, "Calibration function", "Fitted on the validation period 07-12.2016")
show_fig(fig, "calibrator")

# %% 6. Logistic regression - coefficients and odds ratios
pre, clf = lr.named_steps["pre"], lr.named_steps["lr"]
names = pre.get_feature_names_out()
coef = pd.DataFrame({"Variable": names, "Coefficient": clf.coef_[0]})
coef["Odds ratio (OR)"] = np.exp(coef["Coefficient"])
coef["Type"] = np.where(coef["Variable"].str.startswith("num__"), "numeric (per 1 std. dev.)", "category (vs. the rest)")
coef["Variable"] = (coef["Variable"].str.replace("num__", "", regex=False)
                   .str.replace("cat__", "", regex=False))
coef = coef.reindex(coef["Coefficient"].abs().sort_values(ascending=False).index)
header("LOGISTIC REGRESSION")
print(f"Intercept: {clf.intercept_[0]:+.3f}   |   variables after encoding: {len(names)}   |   C={clf.C}")
print("Reading the OR: >1 increases the odds of no-arrival, <1 decreases them.")
print(coef.head(25).round(3).to_string(index=False))
coef.to_csv(OUT / "logistic_coefficients.csv", index=False, encoding="utf-8-sig")

fig, ax = plt.subplots(figsize=(9, 7))
top = coef.head(20).iloc[::-1]
ax.barh(top["Variable"], top["Coefficient"],
        color=[COL["critical"] if v > 0 else COL["s1"] for v in top["Coefficient"]], height=0.65)
ax.axvline(0, color=COL["neutral"], lw=1)
ax.grid(axis="y", visible=False)
titles(ax, "Logistic regression - 20 largest coefficients",
       "Red increases the risk of no-arrival, blue decreases it (log-odds)")
fig.tight_layout()
show_fig(fig, "logistic_coefficients")

# %% 7. Segment heuristic (the "current practice" benchmark)
header("SEGMENT HEURISTIC - historical no-arrival rate per segment")
tab = seg.table_.rename("Probability").reset_index().sort_values("Probability", ascending=False)
print(f"Number of segments: {len(tab)}, global mean: {seg.global_:.3f}")
print(tab.head(15).round(3).to_string(index=False))

# %% 8. One booking in three models + LightGBM explanation (SHAP)
example = te[(te["days_to_arrival"] == 14)].sort_values("p_lgbm", ascending=False).iloc[[0, len(te[te["days_to_arrival"] == 14]) // 2]]
contrib = model.contributions(example)
header("EXAMPLE BOOKINGS (D-14): predictions of all models")
cols = ["booking_id", "hotel", "arrival", "lead_time", "market_segment", "deposit_type", "customer_type",
        "previous_cancellations", "p_segment", "p_logistic", "p_lgbm", "no_arrival"]
print(example[cols].assign(arrival=example["arrival"].dt.strftime("%Y-%m-%d")).round(3).to_string(index=False))
for i, (_, row) in enumerate(example.iterrows()):
    c = contrib.iloc[i].sort_values(key=abs, ascending=False).head(6)
    print(f"\nBooking {row['booking_id']}: 6 largest feature contributions (SHAP, log-odds; + increases risk):")
    for f, v in c.items():
        print(f"   {FEATURE_LABELS.get(f, f):35s} = {str(model.features(example.iloc[[i]])[f].iloc[0]):12s} {v:+.3f}")

# %% 9. Your own hypothetical booking - "what if"
new = pd.DataFrame([{
    "hotel": "City Hotel", "lead_time": 120, "arrival": pd.Timestamp("2017-07-14"),
    "arrival_date_week_number": 28, "stays_in_weekend_nights": 2, "stays_in_week_nights": 1,
    "nights": 3, "adults": 2, "children": 0, "babies": 0, "meal": "BB",
    "market_segment": "Online TA", "distribution_channel": "TA/TO", "is_repeated_guest": 0,
    "previous_cancellations": 0, "previous_bookings_not_canceled": 0, "reserved_room_type": "A",
    "deposit_type": "No Deposit", "agent_cat": "A9", "has_company": 0, "days_in_waiting_list": 0,
    "customer_type": "Transient", "adr": 120.0,
}])
header("HYPOTHETICAL BOOKING - risk on successive days before arrival")
rows = []
for h in [120, 90, 60, 30, 14, 7, 3, 1, 0]:
    r = new.assign(days_to_arrival=h)
    rows.append({"Days to arrival": h,
                 "LightGBM + calibration": float(model.predict(r)[0]),
                 "Logistic regression": float(lr.predict_proba(_lr_frame(make_features(r)))[0, 1]),
                 "Heuristic": float(seg.predict(r)[0])})
print(pd.DataFrame(rows).round(3).to_string(index=False))
print("\nChange the fields in 'new' (e.g. deposit_type='Non Refund', previous_cancellations=1) and re-run the cell.")

print(f"\nCharts and tables saved to: {OUT}")
