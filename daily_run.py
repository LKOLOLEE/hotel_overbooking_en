"""
Daily production run: overbooking plan + list of bookings to contact
+ HTML report ready for online publishing (outputs/site/index.html).

Examples:
    python daily_run.py                       # demo day from config.DEMO_AS_OF
    python daily_run.py --as-of 2017-07-10    # a specific day within the data
    python daily_run.py --replay              # DEMO mode for the scheduler: today's date
                                              #   -> next day of the 2017 test period
    python daily_run.py --as-of today --data D:/pms/todays_export.csv   # production mode
    python daily_run.py --replay --publish-dir \\\\server\\www\\report --open

Outputs:
    outputs/daily/<date>/      overbooking_plan.xlsx, risk_list.xlsx, plan.png, daily_report.html
    outputs/site/              index.html (latest report), archive/, files/  -> to publish
If the models do not exist, they are trained automatically (approx. 1-2 min).
"""
import argparse
import json
import shutil
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use("Agg")

import joblib
import pandas as pd

import config as C
from src import actions
from src import overbooking as ob
from src.daily_report import PLAN_FILE, RISK_FILE, build_daily_html, replay_date, write_site
from src.data import load_clean
from src.eda import HOTEL_NAMES
from src.modeling import CancellationModel
from src.occupancy import last_minute_show_rate, nightly_occupancy, pickup_matrix
from src.overbooking_plots import fig_forward_plan


def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def ensure_models(data_path=None):
    if (C.MODELS / "occupancy_forecaster.joblib").exists() and (C.MODELS / "settings.json").exists():
        return
    log("No models found - training (one-off, approx. 1-2 min) ...")
    from src.pipeline import run_all
    run_all(save_outputs=True, verbose=True)


def main():
    ap = argparse.ArgumentParser(description="Daily overbooking and cancellation-risk report")
    ap.add_argument("--as-of", default=None,
                    help="report date (YYYY-MM-DD) or 'today'; defaults to config.DEMO_AS_OF")
    ap.add_argument("--replay", action="store_true",
                    help="DEMO mode: today's date mapped to the next day of the 2017 test period")
    ap.add_argument("--data", default=None, help="path to the current PMS export (CSV)")
    ap.add_argument("--days", type=int, default=C.FORWARD_DAYS)
    ap.add_argument("--publish-dir", default=None,
                    help="optional: folder to copy the website to (e.g. a web server network share)")
    ap.add_argument("--open", action="store_true", help="open the report in the browser")
    args = ap.parse_args()

    if args.replay:
        as_of = replay_date()
        mode_note = f"DEMO: historical data, day in data {as_of:%d %b %Y}"
    elif args.as_of == "today":
        as_of, mode_note = pd.Timestamp(datetime.now().date()), ""
    else:
        as_of, mode_note = pd.Timestamp(args.as_of or C.DEMO_AS_OF), "DEMO: historical data"
    log(f"Report for {as_of:%Y-%m-%d}")

    ensure_models(args.data)
    model = CancellationModel.load()
    forecaster = joblib.load(C.MODELS / "occupancy_forecaster.joblib")
    settings = json.loads((C.MODELS / "settings.json").read_text(encoding="utf-8"))
    capacity = settings["capacity"]
    thr = settings["contact_threshold"] if settings["contact_threshold"] is not None else float("nan")

    try:
        df = load_clean(args.data)
    except FileNotFoundError as exc:
        sys.exit(f"ERROR: {exc}")
    last = df["arrival"].max()
    if as_of > last or as_of < df["arrival"].min() + pd.Timedelta(days=60):
        sys.exit(f"ERROR: no data for {as_of:%Y-%m-%d} (the data covers arrivals up to {last:%Y-%m-%d}). "
                 "Use --replay (demo mode) or --data with a current PMS export.")
    occ = nightly_occupancy(df)
    picks = pickup_matrix(df)
    q = last_minute_show_rate(df)

    plan = ob.forward_plan(df, model, occ, capacity, picks, q, forecaster, as_of, days=args.days)
    risk, summary = actions.risk_list(df, model, as_of, contact_threshold=thr)

    out = C.DAILY / f"{as_of:%Y-%m-%d}"
    out.mkdir(parents=True, exist_ok=True)
    plan_out = plan.assign(hotel=plan["hotel"].map(HOTEL_NAMES)).drop(columns=["realized"])
    plan_out.to_excel(out / PLAN_FILE, index=False)
    risk.drop(columns=["Did not arrive (verification)"]).to_excel(out / RISK_FILE, index=False)
    fig = fig_forward_plan(plan, as_of)
    fig.savefig(out / "plan.png")
    (out / "daily_report.html").write_text(
        build_daily_html(plan, risk, summary, as_of, fig, mode_note,
                         downloads={"Overbooking plan (Excel)": PLAN_FILE,
                                    "Bookings to contact (Excel)": RISK_FILE}),
        encoding="utf-8")
    index = write_site(plan, risk, summary, as_of, fig, out, mode_note)

    if args.publish_dir:
        dest = Path(args.publish_dir)
        shutil.copytree(C.SITE, dest, dirs_exist_ok=True)
        log(f"Website copied to: {dest}")

    log("Summary:")
    for k, v in summary.items():
        print(f"    {k:50s} {v:,.1f}" if isinstance(v, float) else f"    {k:50s} {v}")
    log(f"Report: {index}")
    if args.open:
        webbrowser.open(index.resolve().as_uri())


if __name__ == "__main__":
    main()
