# Cancellation Forecasting & Smart Overbooking

**Hotel revenue management · Data Science proof of concept**

Booking-level prediction of cancellations and no-shows, translated into a nightly overbooking
recommendation and a daily list of bookings to contact before they cancel.

---

## Business problem

| | City hotel | Resort |
|---|---|---|
| Bookings cancelled | 41% | 27% |
| Of which in the last 7 days before arrival | 12% | 12% |
| Revenue cancelled late (≤ 7 days) | €1.17m | €0.37m |

Hotels usually offset cancellations with a single, fixed overbooking percentage. On high-demand
nights this leaves rooms empty, and on others it leads to walked guests. Nobody knows **which**
bookings are at risk, so nobody acts on them.

## Approach

1. **Risk model.** LightGBM with isotonic calibration scores each active booking at any point from D-180 to D-0, using only information available on that day (leakage-free snapshots).
2. **Occupancy forecast.** Booking-level probabilities are aggregated into a distribution of final occupancy per night (Poisson-binomial + pickup), then corrected by a stage-2 booking-curve model with conformal intervals.
3. **Overbooking optimisation.** For each night the extra bookings to accept are chosen by trading off the value of a room (ADR + ancillary) against the cost of walking a guest.
4. **Preventive actions.** High-risk bookings come with their top SHAP drivers and a recommended action. The contact threshold is set from an A/B uplift analysis.

## Key results (out-of-time test, Jan–Aug 2017)

| Metric | Current practice | This solution |
|---|---|---|
| ROC AUC (booking level) | 0.783 (segment heuristic) | **0.802** |
| Brier score | 0.0785 | **0.0754** |
| Overbooking net result, 2 hotels | −€112k (static 5%) | **+€68k** (37% of the oracle upper bound) |
| Guests walked | 697 | **205** |
| Annualised net benefit¹ | – | **≈ €118k**, payback ≈ 3 months |

¹ Includes synthetic assumptions (walk cost, turned-away demand, effect of guest contact), all listed in `config.py`.
Validate them on client data before making commitments.

---

## Quick start

```bash
python -m venv .venv && .venv\Scripts\activate      # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py                               # ~1-2 min, opens the client report
python daily_run.py --replay --open                  # daily operational report (demo mode)
```

PyCharm setup, scheduling and model inspection are covered in **[docs/operations.md](docs/operations.md)**.

## Deliverables

| Output | Audience |
|---|---|
| `outputs/reports/client_report.html` | Management: findings, model, backtest, business case, assumptions |
| `outputs/reports/project_results.xlsx` | Analysts: all tables, night-by-night backtest |
| `outputs/site/index.html` (+ archive) | Revenue & front office: daily plan and contact list, ready to publish |
| `notebooks/client_report.ipynb` | Walkthrough of the full analysis |
| `inspect_models.py` | Data science: trees, coefficients, calibration, SHAP, what-if |

## Repository layout

```
config.py            parameters and all [SYNTHETIC] assumptions
run_pipeline.py      end-to-end training, evaluation and reporting
daily_run.py         daily scoring → plan, contact list, website
inspect_models.py    model inspection (cell-by-cell)
src/                 data · features · modeling · occupancy · overbooking · outreach · policy · monitoring · report
notebooks/           client notebook (+ Spyder/PyCharm cell version)
scripts/ .github/ azure-pipelines-*.yml   scheduling (Windows, GitHub Actions, Azure DevOps)
tests/               unit tests (pytest)
```

## Methodology notes

- **Validation is strictly out-of-time.** Train on arrivals up to 06/2016, calibrate on 07–12/2016, test on 01–08/2017.
- **Leakage controls.** `country` is excluded (filled in at check-in in this dataset), as are all fields finalised after booking. ADR and arrival month are excluded too: they caused season and price overfitting and cut test AUC from 0.802 to 0.787.
- **Baselines.** The models are compared with a segment heuristic (current practice), logistic regression and an oracle upper bound. The backtest uses common random numbers.
- **Monitoring.** Population Stability Index (PSI) and monthly calibration are tracked. Agent mix drift is already visible in 2017 (PSI 0.75), so monthly retraining is recommended.

## Data

[Hotel Booking Demand](https://doi.org/10.1016/j.dib.2018.11.126) (Antonio, Almeida & Nunes, 2019; CC BY 4.0):
119,390 bookings from two Portuguese hotels, arrivals 07/2015–08/2017. Elements missing from the
public data (room count, walk cost, turned-away demand, outreach effect, prepayment response) are
synthetic and fully parameterised in `config.py`.

## Limitations & next steps

- **Scope:** two hotels, hotel-level rather than room-type level, and historical data only. The next step is a pilot on client PMS data with real capacities and costs.
- **Synthetic elements:** replace them with a measured walk cost, booking-engine denials and a live A/B test of guest outreach.
- **Annualisation:** the backtest covers January–August, so the annual figure may overstate the effect. Confirm it over a full season.
- **Scale-up:** a central model with hotel-level features for a multi-property rollout, plus a Power BI or Snowflake integration for daily outputs.
