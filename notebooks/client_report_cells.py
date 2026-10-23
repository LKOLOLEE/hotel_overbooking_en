"""Client report - cell version (Spyder / PyCharm / VS Code).
Run the cells in order (Spyder: Ctrl+Enter / Shift+Enter).
Charts appear in the Plots pane, tables in the IPython console."""

# %% [markdown]
# # Cancellation forecasting and smart overbooking
# ### Client report (public *Hotel Booking Demand* data + synthetic elements)
#
# This notebook runs **the whole project** and shows the results intended for the client: diagnosis of the problem,
# model quality, the overbooking backtest, the plan for the coming weeks, the list of bookings to contact,
# cancellation-policy analysis, monitoring and the business case.
#
# > Items marked **SYNTHETIC** (walk cost, turned-away demand on sold-out nights,
# > effect of guest contact, effect of prepayment) are assumptions - all of them live in `config.py`.
#
# Run time: approx. 1-2 minutes.

# %%
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(ROOT))

import pandas as pd
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 200)

import config as C
from src.pipeline import run_all
from src.report import show, write_html, write_excel

R = run_all(as_of=C.DEMO_AS_OF)      # full pipeline: data -> models -> backtest -> plan -> report
T, F = R["tables"], R["figures"]

# %% [markdown]
# ## 1. Executive summary

# %%
show(T["kpi"])
show(F["business"])
show(T["business_items"])

# %% [markdown]
# ## 2. Diagnosis: scale of the cancellation problem
# How many bookings never turn into a stay, when cancellations happen and how often the hotels are full.

# %%
show(T["eda_summary"], decimals={"Bookings": 0, "Avg. booking lead time (days)": 0}, transpose=True)
show(F["eda_trend"])
show(F["eda_lead"])
show(F["eda_segment"])
show(F["eda_timing"])
show(F["eda_occupancy"])

# %% [markdown]
# ## 3. No-arrival risk model
# The model scores risk on a specific day before arrival (D-0 ... D-180) using only information
# available on that day (**leakage-free snapshots**). Time-based validation:
# training 07.2015-06.2016, calibration 07-12.2016, test 01-08.2017.

# %%
show(T["model_comparison"], decimals={"ROC_AUC": 3, "PR_AUC": 3, "Brier": 4, "LogLoss": 4})
show(F["model_roc"])
show(F["model_calibration"])
show(F["model_auc_horizon"])
show(T["model_by_horizon"], decimals={"ROC AUC (heuristic)": 3, "ROC AUC (LightGBM)": 3,
                                      "Brier (heuristic)": 4, "Brier (LightGBM)": 4})

# %% [markdown]
# ## 4. Why guests do not arrive (SHAP)
# SHAP values computed natively by LightGBM. Note: the high risk for the *Non Refund* deposit
# is a known artefact of this dataset (such bookings are almost always cancelled).

# %%
show(F["shap_importance"])
show(F["shap_categories"])
show(F["shap_dependence"])

# %% [markdown]
# ## 5. Occupancy forecast
# The structural forecast (bookings x probability of arrival + expected new bookings)
# is corrected by a stage-2 model; uncertainty intervals come from cross-validation residuals.

# %%
show(F["forecast_accuracy"])
show(T["forecast_accuracy"], decimals={"Nights": 0})

# %% [markdown]
# ## 6. Overbooking - 2017 backtest
# For each night the decision was taken at D-3, based solely on what was known at the time.
# Comparison: current practice, static 5% overbooking, ML model, oracle (upper bound).

# %%
show(F["ob_example"])
show(F["ob_cumulative"])
show(F["ob_bars"])
show(T["backtest_summary"])

# %% [markdown]
# ## 7. Overbooking plan for the next 60 days (production mode)

# %%
show(F["forward_plan"])
plan = T["forward_plan"][["hotel", "night", "horizon", "capacity", "in_house", "on_books", "forecast",
                          "occ_p10", "occ_p90", "p_full", "x_opt", "exp_gain", "recommendation"]]
show(plan, max_rows=20)

# %% [markdown]
# ## 8. High-risk bookings (arrival within 14 days)

# %%
show(T["risk_summary"])
show(T["risk_list"], max_rows=25, decimals={"Booking ID (pseudonym)": 0, "Days to arrival": 0, "Nights": 0})

# %% [markdown]
# ## 9. Preventive actions - A/B test of guest contact  *(SYNTHETIC)*
# The effect of contact was applied synthetically to real bookings to demonstrate the evaluation method:
# difference in rates with a confidence interval, effect by risk decile and choosing the group worth contacting.

# %%
show(T["outreach_overall"])
show(F["outreach_uplift"])
show(T["outreach_deciles"])

# %% [markdown]
# ## 10. Cancellation and prepayment policies

# %%
show(F["policy_heat"])
show(F["policy_whatif"])
show(T["policy_whatif"][["Hotel", "Channel", "Lead time (days)", "Bookings", "Cancellation rate",
                         "Late cancellation rate", "Revenue change per year [EUR]", "Recommendation"]], max_rows=12)
show(T["channels"])

# %% [markdown]
# ## 11. Model monitoring (data drift and stability)

# %%
show(F["drift"])
show(T["drift"], decimals={"PSI": 3}, max_rows=15)
show(F["monthly"])
show(T["monthly"], decimals={"ROC AUC": 3, "Brier": 4})

# %% [markdown]
# ## 12. Assumptions and limitations

# %%
show(T["assumptions"])

# %% [markdown]
# ## 13. Export: HTML report and Excel
# The HTML report (a single file with charts) can be sent to the client; the Excel file contains the full tables.

# %%
html_path = write_html(R)
xlsx_path = write_excel(R)
print(html_path)
print(xlsx_path)
# import webbrowser; webbrowser.open(html_path.resolve().as_uri())   # open in the browser
