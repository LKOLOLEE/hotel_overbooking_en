"""Risk models: segment heuristic, logistic regression, LightGBM + calibration."""
from __future__ import annotations

import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config as C
from src.data import CATEGORICAL
from src.features import NUMERIC_FEATURES, FEATURES, category_levels, make_features


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def expected_calibration_error(y, p, bins=10) -> float:
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def metrics(y, p) -> dict:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "ROC_AUC": roc_auc_score(y, p),
        "PR_AUC": average_precision_score(y, p),
        "Brier": brier_score_loss(y, p),
        "LogLoss": log_loss(y, p),
        "ECE": expected_calibration_error(y, p),
        "Mean predicted": float(np.mean(p)),
        "Mean actual": float(np.mean(y)),
    }


# ---------------------------------------------------------------------------
# Baseline 1: "how a revenue manager does it" heuristic - historical
# cancellation rate in the segment (hotel x segment x deposit x horizon bucket)
# ---------------------------------------------------------------------------
def _horizon_bucket(dta):
    return pd.cut(np.asarray(dta, dtype=float), [-1, 0, 3, 14, 60, 10_000],
                  labels=["D-0", "D1-3", "D4-14", "D15-60", "D60+"]).astype(str)


class SegmentRateModel:
    keys = ["hotel", "market_segment", "deposit_type", "hb"]

    def fit(self, snap: pd.DataFrame):
        d = snap[["hotel", "market_segment", "deposit_type"]].copy()
        d["hb"] = _horizon_bucket(snap["days_to_arrival"])
        d["y"] = snap["no_arrival"].values
        self.global_ = d["y"].mean()
        g = d.groupby(self.keys, observed=True)["y"].agg(["sum", "count"])
        # shrinkage towards the global mean
        self.table_ = ((g["sum"] + 20 * self.global_) / (g["count"] + 20)).rename("p")
        return self

    def predict(self, snap: pd.DataFrame) -> np.ndarray:
        d = snap[["hotel", "market_segment", "deposit_type"]].copy()
        d["hb"] = _horizon_bucket(snap["days_to_arrival"])
        out = d.join(self.table_, on=self.keys)["p"].fillna(self.global_)
        return out.to_numpy()


# ---------------------------------------------------------------------------
# Baseline 2: logistic regression
# ---------------------------------------------------------------------------
def train_logistic(X_train: pd.DataFrame, y_train) -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), CATEGORICAL),
    ])
    model = Pipeline([("pre", pre),
                      ("lr", LogisticRegression(max_iter=1000, C=1.0))])
    model.fit(_lr_frame(X_train), y_train)
    return model


def _lr_frame(X: pd.DataFrame) -> pd.DataFrame:
    Z = X.copy()
    for c in CATEGORICAL:
        Z[c] = Z[c].astype(str)
    return Z


# ---------------------------------------------------------------------------
# Main model: LightGBM + isotonic calibration
# ---------------------------------------------------------------------------
class CancellationModel:
    """Wrapper: features -> LightGBM -> calibration. Can be saved to disk."""

    def __init__(self):
        self.booster = None
        self.calibrator = None
        self.categories = None

    # -- training -----------------------------------------------------------
    def fit(self, snap_train: pd.DataFrame, snap_valid: pd.DataFrame):
        self.categories = category_levels(snap_train)
        Xtr = make_features(snap_train, categories=self.categories)
        Xva = make_features(snap_valid, categories=self.categories)
        dtr = lgb.Dataset(Xtr, snap_train["no_arrival"], categorical_feature=CATEGORICAL)
        dva = lgb.Dataset(Xva, snap_valid["no_arrival"], reference=dtr,
                          categorical_feature=CATEGORICAL)
        self.evals_ = {}
        self.booster = lgb.train(
            C.LGBM_PARAMS, dtr, num_boost_round=C.LGBM_NUM_ROUNDS,
            valid_sets=[dtr, dva], valid_names=["train", "valid"],
            callbacks=[lgb.early_stopping(C.LGBM_EARLY_STOPPING, verbose=False),
                       lgb.record_evaluation(self.evals_)],
        )
        raw_valid = self.booster.predict(Xva, num_iteration=self.booster.best_iteration)
        self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.calibrator.fit(raw_valid, snap_valid["no_arrival"])
        return self

    # -- prediction ---------------------------------------------------------
    def features(self, rows, days_to_arrival=None):
        return make_features(rows, days_to_arrival, categories=self.categories)

    def predict_raw(self, rows, days_to_arrival=None) -> np.ndarray:
        X = self.features(rows, days_to_arrival)
        return self.booster.predict(X, num_iteration=self.booster.best_iteration)

    def predict(self, rows, days_to_arrival=None) -> np.ndarray:
        """Calibrated probability that the guest will NOT arrive."""
        return self.calibrator.predict(self.predict_raw(rows, days_to_arrival))

    def contributions(self, rows, days_to_arrival=None) -> pd.DataFrame:
        """SHAP values (native LightGBM, on the log-odds scale)."""
        X = self.features(rows, days_to_arrival)
        contrib = self.booster.predict(X, num_iteration=self.booster.best_iteration,
                                       pred_contrib=True)
        return pd.DataFrame(contrib[:, :-1], columns=FEATURES, index=rows.index)

    # -- save / load ----------------------------------------------------------
    def save(self, folder=C.MODELS):
        folder.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(folder / "lgbm_model.txt"),
                                num_iteration=self.booster.best_iteration)
        joblib.dump(self.calibrator, folder / "calibrator.joblib")
        (folder / "categories.json").write_text(
            json.dumps(self.categories, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, folder=C.MODELS):
        m = cls()
        m.booster = lgb.Booster(model_file=str(folder / "lgbm_model.txt"))
        m.calibrator = joblib.load(folder / "calibrator.joblib")
        m.categories = json.loads((folder / "categories.json").read_text(encoding="utf-8"))
        return m


# ---------------------------------------------------------------------------
# Training and comparison of all models
# ---------------------------------------------------------------------------
def train_and_evaluate(snap: pd.DataFrame, verbose=True):
    tr = snap[snap["split"] == "train"]
    va = snap[snap["split"] == "valid"]
    te = snap[snap["split"] == "test"].copy()

    if verbose:
        print(f"  snapshots: train={len(tr):,}  valid={len(va):,}  test={len(te):,}")

    seg = SegmentRateModel().fit(pd.concat([tr, va]))
    if verbose:
        print("  training logistic regression ...")
    lr = train_logistic(make_features(tr), tr["no_arrival"])
    if verbose:
        print("  training LightGBM ...")
    model = CancellationModel().fit(tr, va)

    te["p_segment"] = seg.predict(te)
    te["p_logistic"] = lr.predict_proba(_lr_frame(make_features(te)))[:, 1]
    te["p_lgbm_raw"] = model.predict_raw(te)
    te["p_lgbm"] = model.predict(te)

    names = {
        "p_segment": "Segment heuristic (current practice)",
        "p_logistic": "Logistic regression",
        "p_lgbm_raw": "LightGBM (uncalibrated)",
        "p_lgbm": "LightGBM + calibration (target model)",
    }
    comp = pd.DataFrame({names[c]: metrics(te["no_arrival"], te[c]) for c in names}).T

    rows = []
    for h, g in te.groupby("days_to_arrival"):
        rows.append({
            "Days to arrival (D-h)": h,
            "Bookings": len(g),
            "No-arrival rate": g["no_arrival"].mean(),
            "ROC AUC (heuristic)": roc_auc_score(g["no_arrival"], g["p_segment"]),
            "ROC AUC (LightGBM)": roc_auc_score(g["no_arrival"], g["p_lgbm"]),
            "Brier (heuristic)": brier_score_loss(g["no_arrival"], g["p_segment"]),
            "Brier (LightGBM)": brier_score_loss(g["no_arrival"], g["p_lgbm"]),
        })
    by_h = pd.DataFrame(rows).set_index("Days to arrival (D-h)")
    # save the benchmark models (for inspection: inspect_models.py)
    C.MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(lr, C.MODELS / "logistic_regression.joblib")
    joblib.dump(seg, C.MODELS / "segment_heuristic.joblib")
    return model, te, comp, by_h
