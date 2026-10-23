# Cancellation forecasting and smart overbooking

A complete Data Science project for a hotel (or a hotel chain). It covers:
- a cancellation / no-show risk model;
- a forecast of actual occupancy;
- overbooking optimisation;
- a list of bookings to contact;
- prepayment-policy analysis;
- model monitoring;
- a business case.

All client-facing results are generated in Python, as a Jupyter notebook, an HTML report (a single file with charts) and an Excel workbook.

**Data:** the public **Hotel Booking Demand** dataset (Antonio, Almeida, Nunes, *Data in Brief* 2019, CC BY 4.0 licence). It holds 119,390 bookings from a city hotel and a resort in Portugal, with arrivals from 07.2015 to 08.2017. The file is included in `data/raw/`. If you delete it, the script downloads it again.

---

## Quick start

```bash
cd hotel_overbooking_en
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python run_pipeline.py          # approx. 1-2 min; opens the HTML report in the browser at the end
```

Requires Python 3.9+. Tested on Python 3.13, pandas 3.0 and LightGBM 4.7.

| Environment | How to run |
|---|---|
| **Jupyter Notebook** | `jupyter notebook notebooks/client_report.ipynb`, then *Run All* |
| **Spyder** | Open `notebooks/client_report_cells.py`, set the working directory to the project folder and run the cells with Shift+Enter. Charts appear in the *Plots* pane. |
| **PyCharm** | Open the project folder. Ready-made run configurations appear in the top-right corner (details below). |
| **Terminal** | `python run_pipeline.py --as-of 2017-07-01` |

**Daily report** (overbooking plan + list of bookings to contact + HTML page):

```bash
python daily_run.py --as-of 2017-07-10 --open   # a specific day within the data
python daily_run.py --replay --open             # DEMO mode for the scheduler (see below)
```

It creates two outputs:
- `outputs/daily/<date>/`: Excel files, a PNG chart and `daily_report.html`;
- the website `outputs/site/`: `index.html` (latest report), `archive/` and `files/`, ready for online publishing.

If the models do not exist yet, `daily_run.py` trains them automatically.

Tests: `python -m pytest -q`

---

## PyCharm

1. **File → Open**, select the `hotel_overbooking_en` folder (the one with `README.md`), then *Trust Project*.
2. **Interpreter.** PyCharm usually detects `.venv` by itself. If it does not, go to *Settings → Project → Python Interpreter → Add Interpreter → Add Local Interpreter → Select existing* and pick `hotel_overbooking_en\.venv\Scripts\python.exe`.
3. The project folder is the content root, so the `config` and `src.*` imports work without extra settings.
4. The *Run/Debug Configurations* list in the top-right corner holds ready-made configurations from the `.run/` folder:
   - `1. Pipeline - training and report`
   - `2. Daily report - demo (replay)`
   - `3. Daily report - chosen date`
   - `4. Tests`
   - `5. Inspect models (LightGBM, LogReg)`

   If they do not appear, right-click `run_pipeline.py` and choose *Run*.
5. `notebooks/client_report.ipynb` opens in PyCharm if your edition supports Jupyter. Otherwise use the browser (`jupyter notebook`) or configuration 1 (HTML report).

---

## Automation: daily report at 08:00 and online publishing

### Important: demo vs production mode
The public data ends on 31 Aug 2017, so "today's" date is not in it. The scheduler therefore uses **`--replay`** mode, which maps each real calendar day to the next day of the test period (15 Jan - 2 Jul 2017, in a loop). The report moves forward one day every day, just like in a real hotel, and the page carries a *DEMO* badge.

In production, use this instead of `--replay`:
```bash
python daily_run.py --as-of today --data <path to today's PMS export>.csv
```
The export must use the same column format as `hotel_bookings.csv`. Retrain the models once a month with `run_pipeline.py`.

### Option A - Windows Task Scheduler (local computer or server)
```bat
powershell -ExecutionPolicy Bypass -File scripts\register_schedule.ps1            :: daily at 08:00
powershell -ExecutionPolicy Bypass -File scripts\register_schedule.ps1 -Time 07:30
```
- The task runs `scripts\daily_run.bat` and logs to `logs\daily_run.log`.
- If the computer was off at 08:00, the report is generated right after it is switched on.
- To remove the task: `scripts\remove_schedule.ps1`.
- To publish, run `daily_run.bat --publish-dir "\\server\www\report"`. It copies the website to a web-server folder (IIS / nginx).

Note: OneDrive and SharePoint do not display HTML files as pages; they download them.

### Option B - GitHub Actions + GitHub Pages (no computer of your own; demo)
1. Push the project to a GitHub repository. `.gitignore` skips `.venv` and `outputs`.
2. Go to *Settings → Pages → Source: GitHub Actions*.
3. The `.github/workflows/daily-report.yml` workflow runs every day at 06:00 UTC (08:00 in summer, 07:00 in winter in Central Europe). To run it manually: *Actions → Run workflow*.
4. The site address is `https://<login>.github.io/<repo>/`.

Limitations:
- GitHub may delay a scheduled run by several minutes.
- GitHub disables schedules in public repositories after 60 days without activity.
- **A GitHub Pages site is public.** Use it only for demos on public data.

### Option C - Azure DevOps + Azure (production with access control)
`azure-pipelines-daily-report.yml` runs at 06:00 UTC, caches the models and publishes the website to the `$web` container of an Azure Storage account (static website). Fill in `STORAGE_ACCOUNT` and `AZURE_SERVICE_CONNECTION`.

For client data that requires sign-in, do not use a public static website. Use **Azure Static Web Apps** with Microsoft Entra ID authentication, so that only named people have access.

### Option D - Power BI
`overbooking_plan.xlsx` and `risk_list.xlsx` (or the same tables written to a database / Snowflake) can feed a Power BI report with scheduled refresh. The client then views it in the Power BI Service with their own permissions.

---

## Inspecting the trained models

Run `inspect_models.py` cell by cell (`# %%`: PyCharm *Run Cell* / Spyder Shift+Enter) or as a whole with `python inspect_models.py`:

| Cell | What it shows |
|---|---|
| 1 | Model files and a metric comparison of the 4 models on 2017 |
| 2-4 | **LightGBM**: parameters, number of trees, encoded categories, feature importance (gain/split), the first tree as text |
| 5 | **Calibrator** (isotonic regression): raw score → probability |
| 6 | **Logistic regression**: coefficients and odds ratios (OR) |
| 7 | **Segment heuristic**: table of no-arrival rates per segment |
| 8 | The same booking in three models + SHAP explanation |
| 9 | A hypothetical "what if" booking: change the fields and re-run |

Charts and tables go to `outputs/inspect/`. Model files in `outputs/models/`:

| File | Contents |
|---|---|
| `lgbm_model.txt` | Text dump of all trees; you can open it in PyCharm |
| `calibrator.joblib` | Isotonic calibrator |
| `logistic_regression.joblib` | scikit-learn pipeline: scaling + one-hot + LogisticRegression |
| `segment_heuristic.joblib` | Segment heuristic |
| `occupancy_forecaster.joblib` | Stage-2 occupancy model |

In code, load them with `CancellationModel.load()` and `joblib.load(...)`.

---

## Outputs (where to look)

| File | Contents |
|---|---|
| `outputs/reports/client_report.html` | Client report: summary, diagnosis, model, backtest, plan, risk list, policies, monitoring, business case, assumptions |
| `outputs/reports/project_results.xlsx` | All tables (60-day plan, risk list, night-by-night backtest, metrics…) |
| `outputs/figures/*.png` | Charts |
| `outputs/tables/*.csv` | Tables as CSV |
| `outputs/models/` | LightGBM model, calibrator, occupancy model, settings |
| `outputs/site/index.html` | Latest daily report (+ `archive/`, `files/`); the website to publish |
| `outputs/daily/<date>/` | Daily report: Excel files, chart, `daily_report.html` |

---

## Project structure

```
hotel_overbooking_en/
├── config.py                 # ALL parameters and synthetic assumptions (marked [SYNTHETIC])
├── run_pipeline.py           # full run + HTML/Excel report
├── inspect_models.py         # model inspection: LightGBM, calibration, logistic regression, heuristic
├── daily_run.py              # daily report: plan + risk list + HTML website (outputs/site)
├── requirements-core.txt     # libraries needed on the server / in the scheduler
├── scripts/                  # Windows Task Scheduler (.bat, register/remove .ps1)
├── .github/workflows/        # GitHub Actions: daily report -> GitHub Pages
├── azure-pipelines-daily-report.yml   # Azure DevOps: daily report -> Azure Storage
├── .run/                     # ready-made PyCharm run configurations
├── data/raw/hotel_bookings.csv
├── notebooks/
│   ├── client_report.ipynb         # client report (Jupyter)
│   └── client_report_cells.py      # the same in # %% cells (Spyder/PyCharm)
├── src/
│   ├── data.py               # loading, cleaning, time-based split
│   ├── features.py           # leakage-free snapshots + features
│   ├── modeling.py           # heuristic, logistic regression, LightGBM + calibration
│   ├── explain.py            # model evaluation, SHAP, risk reasons per booking
│   ├── occupancy.py          # nightly occupancy, capacity, pickup
│   ├── overbooking.py        # occupancy forecast (2 stages), optimisation, backtest, plan
│   ├── overbooking_plots.py
│   ├── actions.py            # daily list of bookings to contact
│   ├── outreach.py           # A/B test of preventive actions [SYNTHETIC] + uplift
│   ├── policy.py             # segments, risk map, prepayment what-if
│   ├── monitoring.py         # PSI, stability of quality over time
│   ├── business_case.py      # ROI, payback period
│   ├── eda.py, plotting.py   # diagnosis and chart style
│   ├── report.py             # HTML report, Excel, display in notebooks
│   ├── daily_report.py       # daily HTML report, website with archive, replay mode
│   └── pipeline.py           # orchestration
└── tests/test_core.py
```

---

## Methodology in brief

**1. Leakage-free snapshots.** For every booking we take "pictures" of its state at D-180, D-120, …, D-1 and D-0 before arrival. A picture is taken only if the booking was active at that point: already made and not yet cancelled. The cancellation date comes from `reservation_status_date`. The target is whether the guest ultimately did not arrive (a cancellation after that date, or a no-show).

**2. Excluded features (leakage / drift).**
- Leakage:
  - `country`, which this dataset fills in at check-in, so cancelled bookings show "PRT" unnaturally often;
  - `booking_changes`, `assigned_room_type`, `required_car_parking_spaces` and `total_of_special_requests`, which are final values.
- Drift: price (ADR) and arrival month/week. With one year of history the model memorised the season and the price level. Removing them raised the 2017 ROC AUC from about 0.787 to about 0.802.

**3. Time-based validation.**
- Training: arrivals up to 06.2016.
- Validation and calibration: 07-12.2016.
- Test: 01-08.2017.

The model is compared with a segment heuristic (what a revenue manager does today) and with logistic regression. Metrics: ROC AUC, PR AUC, Brier, LogLoss and calibration error (ECE).

**4. Occupancy forecast (2 stages).**
- Structural stage: guests in house + Σ(1 − p) over bookings on the books + pickup as a share of free capacity ("constrained pickup"), with a Poisson-binomial distribution.
- Stage-2 correction: a LightGBM model trained on the validation period. The cancellation model works out-of-sample there; on the training period its AUC is ~0.93, so it would be overfitted.
- Uncertainty: empirical residuals from cross-validation in blocks of weeks (a conformal approach).

**5. Overbooking optimisation.** For each night, choose how many *X* extra bookings to sell to maximise `v·E[min(R+Y, Cap)] − w·E[max(R+Y−Cap, 0)]`, where:
- `R` is occupancy under the current practice;
- `Y ~ Bin(X, q)`, with `q` the arrival rate of last-minute bookings;
- `v` is ADR + ancillary spend;
- `w` is the cost of walking a guest.

**6. 2017 backtest.** The decision is taken at D-3 for every night. Four policies are compared: no change, static 5% overbooking, the ML model, and an oracle (perfect knowledge, i.e. the upper bound). Common random numbers keep the comparison fair.

**7. Preventive actions.** A list of high-risk bookings, each with its 3 main reasons (SHAP) and a recommended action. The contact threshold comes from the (synthetic) A/B test and an uplift model.

---

## Synthetic data - what and why

The public dataset does not contain everything a deployment needs. The missing pieces are **synthetic** and fully controlled from `config.py`:

| Missing from the data | Solution |
|---|---|
| Room count | Maximum realised occupancy (the data clearly hits a ceiling); override with `CAPACITY_OVERRIDE` |
| Turned-away demand on sold-out nights | Poisson(10% of capacity) on nights with occupancy ≥ 95% |
| Cost of walking a guest | 2 × ADR + €150 |
| Ancillary guest spend | €35 (resort) / €20 (city hotel) per room-night |
| Effect of guest contact | Synthetic A/B test on real bookings (max −30% cancellations for "undecided" guests) |
| Effect of a prepayment requirement | −40% cancellations, −8% bookings; 80% resale of early cancellations |
| Project costs | €30k implementation, €6k per year running cost |

In a real deployment these come from:
- the room count: the PMS;
- the walk cost: finance;
- turned-away demand: the booking engine (requests with no availability);
- the effect of contact: a real A/B test.

---

## Adapting to the hotel's data

1. Replace `data/raw/hotel_bookings.csv` with a PMS export that has the same columns, or add a mapping in `src/data.py::clean`. The key fields are booking date, arrival date, length of stay, status, **cancellation date**, channel, segment, deposit, room type and rate.
2. In `config.py`, set `CAPACITY_OVERRIDE`, the costs and the split dates.
3. Run `python run_pipeline.py` and check the report. After that, run `python daily_run.py --as-of today --data <export>.csv` every day.

## Limitations

- Only 2 hotels in Portugal, 2015-2017. Results must be confirmed on the client's data.
- The analysis is at hotel level, not room type; there is no data on room-type capacity or upgrades.
- The backtest result (January-August) is annualised proportionally, which may overstate the effect.
- The data drifts. The 2017 agent/intermediary mix differs clearly from the training period (PSI > 0.25), and calibration drifts in some months. In production, retrain and recalibrate monthly.
