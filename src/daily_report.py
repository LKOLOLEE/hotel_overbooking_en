"""Daily HTML report for the client + static website (index + archive) for online publishing."""
from __future__ import annotations

import html
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

import config as C
from src.eda import HOTEL_NAMES
from src.report import CSS, _tw, fig_html, fmt_eur, fmt_pct, kpi_tile, table_html

EXTRA_CSS = """
.badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:12px;background:#fff6e0;
color:#6b4a00;border:1px solid #eda100;margin-left:8px;vertical-align:middle}
.downloads a{display:inline-block;margin:4px 12px 4px 0;color:var(--accent)}
ul.archive{columns:3;font-size:13.5px} ul.archive a{color:var(--accent);text-decoration:none}
@media(max-width:760px){ul.archive{columns:1}}
"""
PLAN_FILE, RISK_FILE = "overbooking_plan.xlsx", "risk_list.xlsx"


def replay_date(today: pd.Timestamp | None = None) -> pd.Timestamp:
    """Maps a real calendar date onto a day of the test period (demo mode)."""
    today = pd.Timestamp(today or datetime.now().date())
    start, end = pd.Timestamp(C.REPLAY_START), pd.Timestamp(C.REPLAY_END)
    span = (end - start).days + 1
    offset = (today - pd.Timestamp(C.REPLAY_ANCHOR)).days % span
    return start + pd.Timedelta(days=offset)


def _plan_table(plan: pd.DataFrame, days: int = 14) -> pd.DataFrame:
    as_of = plan["night"].min()
    p = plan[plan["night"] < as_of + pd.Timedelta(days=days)]
    return (p.assign(Hotel=p["hotel"].map(HOTEL_NAMES))
             [["night", "Hotel", "capacity", "in_house", "on_books", "forecast",
               "occ_p10", "occ_p90", "p_full", "x_opt", "exp_gain", "recommendation"]]
             .rename(columns={"night": "Night", "capacity": "Capacity", "in_house": "Guests in house",
                              "on_books": "Bookings on the books", "forecast": "Occupancy forecast",
                              "occ_p10": "P10", "occ_p90": "P90", "p_full": "Probability of full hotel",
                              "x_opt": "Extra bookings", "exp_gain": "Expected gain [EUR]",
                              "recommendation": "Recommendation"})
             .sort_values(["Night", "Hotel"]))


def build_daily_html(plan, risk, summary, as_of, fig, mode_note: str = "",
                     archive: list[str] | None = None, downloads: dict | None = None) -> str:
    as_of = pd.Timestamp(as_of)
    week = plan[plan["night"] < as_of + pd.Timedelta(days=7)]
    ob_nights = int(plan["recommendation"].str.startswith("Allow").sum())
    low_nights = int(plan["recommendation"].str.startswith("Low").sum())
    occ_week = (week["forecast"].sum() / week["capacity"].sum()) if len(week) else float("nan")
    tiles = "".join([
        kpi_tile("Forecast occupancy (7 days)", fmt_pct(occ_week, 0), "both hotels, today's forecast"),
        kpi_tile("Nights where overbooking is allowed", str(ob_nights),
                 f"out of {len(plan)} (hotel × night, {plan['night'].nunique()} days)"),
        kpi_tile("Low-occupancy nights", str(low_nights), "candidates for promotion", "bad" if low_nights else ""),
        kpi_tile("Bookings to contact", str(summary["Phone contact"] + summary["E-mail contact"]),
                 f"phone: {summary['Phone contact']}, e-mail: {summary['E-mail contact']}"),
        kpi_tile("Revenue at risk (14 days)", fmt_eur(summary["Revenue at risk [EUR]"]),
                 f"expected no-arrivals: {summary['Expected no-arrivals']:.0f}"),
    ])
    web_cols = ["Booking ID (pseudonym)", "Hotel", "Arrival", "Days to arrival", "Channel",
                "No-arrival risk", "Revenue at risk [EUR]", "Recommended action", "Main risk drivers"]
    rl = risk[[c for c in web_cols if c in risk.columns]]          # all columns are in the Excel file
    dl = ""
    if downloads:
        dl = '<p class="downloads">' + "".join(
            f'<a href="{html.escape(href)}" download>{html.escape(label)}</a>' for label, href in downloads.items()) + "</p>"
    arch = ""
    if archive:
        arch = ('<section id="archive"><h2>Report archive</h2><ul class="archive">'
                + "".join(f'<li><a href="archive/{d}.html">{pd.Timestamp(d):%d %b %Y}</a></li>' for d in archive)
                + "</ul></section>")
    badge = f'<span class="badge">{html.escape(mode_note)}</span>' if mode_note else ""
    body = f"""
<section><h2>Today at a glance</h2><div class="kpis">{tiles}</div>{dl}</section>
<section><h2>Overbooking plan</h2>{fig_html(fig)}
<h3>Next 14 nights</h3>{_tw(table_html(_plan_table(plan), index=False, decimals={"Occupancy forecast": 0}))}</section>
<section><h2>Bookings to contact (arrival within 14 days)</h2>
{_tw(table_html(rl, index=False, max_rows=40, decimals={"Booking ID (pseudonym)": 0, "Days to arrival": 0, "Nights": 0}))}
<p class="lead">The full list is in the Excel file. IDs are pseudonyms.</p></section>
{arch}"""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Daily overbooking report</title><style>{CSS}{EXTRA_CSS}</style></head><body>
<header><div class="wrap"><h1>Daily report: overbooking and cancellation risk {badge}</h1>
<p>As of {as_of:%d %b %Y} · generated {datetime.now():%d %b %Y %H:%M}</p></div></header>
<main>{body}<footer>Model: LightGBM + calibration, stage-2 occupancy forecast. Methodology details are in the project report.</footer>
</main></body></html>"""


def write_site(plan, risk, summary, as_of, fig, daily_dir: Path, mode_note: str = "",
               site_dir: Path = C.SITE) -> Path:
    """Writes outputs/site/index.html (latest report), the archive and downloadable files."""
    as_of = pd.Timestamp(as_of)
    stamp = f"{as_of:%Y-%m-%d}"
    (site_dir / "archive").mkdir(parents=True, exist_ok=True)
    (site_dir / "files").mkdir(parents=True, exist_ok=True)
    for name in [PLAN_FILE, RISK_FILE]:
        shutil.copy2(daily_dir / name, site_dir / "files" / f"{stamp}_{name}")
    downloads = {"Overbooking plan (Excel)": f"files/{stamp}_{PLAN_FILE}",
                 "Bookings to contact (Excel)": f"files/{stamp}_{RISK_FILE}"}

    # archive page (relative links from the sub-folder)
    arch_dl = {k: "../" + v for k, v in downloads.items()}
    (site_dir / "archive" / f"{stamp}.html").write_text(
        build_daily_html(plan, risk, summary, as_of, fig, mode_note, downloads=arch_dl), encoding="utf-8")

    archive = sorted((p.stem for p in (site_dir / "archive").glob("*.html")), reverse=True)[:C.ARCHIVE_DAYS]
    index = site_dir / "index.html"
    index.write_text(build_daily_html(plan, risk, summary, as_of, fig, mode_note,
                                      archive=archive, downloads=downloads), encoding="utf-8")
    (site_dir / ".nojekyll").write_text("", encoding="utf-8")       # GitHub Pages: serve files as-is
    return index
