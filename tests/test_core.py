"""Unit tests of the core logic:  python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import snapshot_mask
from src.monitoring import psi
from src.overbooking import optimize_night, poisson_binomial_pmf


def _toy():
    return pd.DataFrame({
        "lead_time":          [30, 30, 5, 30],
        "is_cancel":          [0, 1, 0, 1],
        "cancel_days_before": [np.nan, 10, np.nan, 40],
    })


def test_snapshot_only_active_bookings():
    d = _toy()
    # D-14: booking 0 active; 1 active (cancelled only at D-10);
    # 2 not made yet (lead 5); 3 already cancelled (D-40)
    assert snapshot_mask(d, 14).tolist() == [True, True, False, False]
    # D-7: booking 1 already cancelled
    assert snapshot_mask(d, 7).tolist() == [True, False, False, False]


def test_poisson_binomial_moments():
    p = np.array([0.1, 0.5, 0.9, 0.3])
    pmf = poisson_binomial_pmf(p)
    k = np.arange(len(pmf))
    assert np.isclose(pmf.sum(), 1)
    assert np.isclose(k @ pmf, p.sum())
    assert np.isclose(((k - p.sum()) ** 2) @ pmf, (p * (1 - p)).sum())


def test_overbooking_extremes():
    pmf = np.zeros(101)
    pmf[90] = 1.0                        # a certain 90 guests, capacity 100
    cheap = optimize_night(pmf, 100, value=100, w_cost=0.0, q=1.0, x_max=20)
    assert cheap["x_opt"] >= 10          # walking is free -> sell
    costly = optimize_night(pmf, 100, value=100, w_cost=1e6, q=1.0, x_max=20)
    assert costly["x_opt"] == 10         # fill the free rooms exactly
    assert costly["p_walk_opt"] == 0


def test_psi_identical_is_zero():
    x = pd.Series(np.random.default_rng(0).normal(size=5000))
    assert psi(x, x) < 1e-6
