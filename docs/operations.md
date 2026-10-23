# Operations guide

## 1. Environment

```bash
python -m venv .venv
.venv\Scripts\activate                 # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt        # requirements-core.txt = runtime only (servers / CI)
```

If the terminal shows `(base)` (Anaconda), run `conda deactivate` before activating `.venv`.
To stop this permanently, run `conda config --set auto_activate_base false`.

## 2. PyCharm

1. **File → Open** the project folder (the one containing `README.md`).
2. **Settings → Python Interpreter → Add Interpreter → Add Local Interpreter**:
   - *Select existing* `.venv\Scripts\python.exe`, or
   - *Generate new* (Virtualenv).

   You can also click the interpreter name in the bottom-right status bar.
3. The ready-made run configurations (from `.run/`) appear in the top-right list:
   - 1. Pipeline
   - 2. Daily report (replay)
   - 3. Daily report (chosen date)
   - 4. Tests
   - 5. Inspect models

Run buttons use the project interpreter, not the terminal session.

## 3. Daily report and scheduling

```bash
python daily_run.py --replay                                  # demo: maps today onto the 2017 test period
python daily_run.py --as-of today --data <pms_export>.csv     # production
python daily_run.py --replay --publish-dir "\\server\www\report"
```

The public data ends in 08/2017, so scheduled demo runs use `--replay` and the page shows a *DEMO* badge.
Models are trained automatically if they are missing. Retrain monthly with `run_pipeline.py`.

| Option | Setup | Notes |
|---|---|---|
| Windows Task Scheduler | `powershell -ExecutionPolicy Bypass -File scripts\register_schedule.ps1 [-Time 07:30]` | Runs at 08:00 and catches up if the computer was off. Logs to `logs\daily_run.log`; remove with `scripts\remove_schedule.ps1` |
| GitHub Actions + Pages | `.github/workflows/daily-report.yml`; Settings → Pages → Source: GitHub Actions | 06:00 UTC. **Public site: demo only** |
| Azure DevOps + Storage | `azure-pipelines-daily-report.yml`; set `STORAGE_ACCOUNT`, `AZURE_SERVICE_CONNECTION` | For client data, use Static Web Apps with Entra ID sign-in |
| Power BI | Point a dataset at `overbooking_plan.xlsx` / `risk_list.xlsx` (or the same tables in Snowflake) | Scheduled refresh, access through the client's own permissions |

## 4. Model inspection

`inspect_models.py`, run cell by cell (`# %%`):

| Cell | Content |
|---|---|
| 1 | Model files and test metrics |
| 2–4 | LightGBM parameters, encoded categories, gain/split importance, first tree as text |
| 5 | Isotonic calibrator |
| 6 | Logistic regression coefficients and odds ratios |
| 7 | Segment heuristic table |
| 8 | One booking scored by all models, with SHAP drivers |
| 9 | What-if scoring of a hypothetical booking |

Model artefacts in `outputs/models/`:

| File | Content |
|---|---|
| `lgbm_model.txt` | All trees, readable as text |
| `calibrator.joblib` | Isotonic calibrator |
| `logistic_regression.joblib` | Logistic regression pipeline |
| `segment_heuristic.joblib` | Segment heuristic table |
| `occupancy_forecaster.joblib` | Stage-2 occupancy forecaster |
| `settings.json` | Run settings |

## 5. Adapting to client data

1. Provide a PMS export with the same columns as `data/raw/hotel_bookings.csv`, or map the columns in `src/data.py::clean`. The required fields are booking date, arrival, nights, status, **cancellation date**, channel, segment, deposit, room type and rate.
2. In `config.py`, set `CAPACITY_OVERRIDE`, the costs and the split dates.
3. Run `run_pipeline.py`, review the report, then schedule `daily_run.py --as-of today --data ...`.
