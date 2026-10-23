"""Client report: HTML (single file, embedded charts) + Excel workbook."""
from __future__ import annotations

import base64
import html
import io
from datetime import datetime

import numpy as np
import pandas as pd

import config as C
from src.eda import HOTEL_NAMES
from src.outreach import THRESHOLD_KEY
from src.overbooking import POLICY_NAMES

PCT_KEYS = ["rate", "Rate", "Share", "No-arrival risk", "Coverage", "WAPE", "ROI", "Uplift",
            "CI lower", "CI upper", "Mean predicted", "Mean actual", "No-arrivals",
            "Calibration error", "% of potential", "Probability", "ECE", "Reduction", "threshold"]


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def _num(x, dec=0):
    if pd.isna(x):
        return "-"
    return f"{x:,.{dec}f}"


def fmt_pct(x, dec=1):
    return "-" if pd.isna(x) else f"{x * 100:.{dec}f}%"


def fmt_eur(x):
    if pd.isna(x):
        return "-"
    return f"-€{_num(abs(x))}" if x < 0 else f"€{_num(x)}"


def format_table(df: pd.DataFrame, decimals: dict | None = None) -> pd.DataFrame:
    decimals = decimals or {}
    out = df.copy()
    for col in out.columns:
        s = out[col]
        name = str(col)
        if s.dtype == object:                      # e.g. a column built from a dict with mixed types
            num = pd.to_numeric(s, errors="coerce")
            if num.notna().sum() == s.notna().sum() and s.notna().any():
                s = num
        if pd.api.types.is_datetime64_any_dtype(s):
            out[col] = s.dt.strftime("%Y-%m-%d")
        elif not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            continue
        elif name in decimals:
            out[col] = s.map(lambda v: _num(v, decimals[name]))
        elif "EUR" in name:
            out[col] = s.map(fmt_eur)
        elif any(k in name for k in PCT_KEYS):
            out[col] = s.map(fmt_pct)
        elif (s.dropna() % 1 == 0).all():
            out[col] = s.map(lambda v: _num(v, 0))
        else:
            out[col] = s.map(lambda v: _num(v, 2))
    return out


def table_html(df: pd.DataFrame, index=True, decimals=None, max_rows=None, transpose=False) -> str:
    d = df.head(max_rows) if max_rows else df
    f = format_table(d, decimals)
    if transpose:
        f = f.T
    return f.to_html(index=index, escape=True, border=0, classes="tbl", na_rep="-")


def fig_html(fig, caption="") -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    b64 = base64.b64encode(buf.getvalue()).decode()
    cap = f"<figcaption>{html.escape(caption)}</figcaption>" if caption else ""
    return f'<figure><img src="data:image/png;base64,{b64}" alt="{html.escape(caption)}">{cap}</figure>'


def kpi_tile(label, value, note="", tone="") -> str:
    return (f'<div class="kpi {tone}"><div class="kpi-label">{html.escape(label)}</div>'
            f'<div class="kpi-value">{html.escape(value)}</div>'
            f'<div class="kpi-note">{html.escape(note)}</div></div>')


CSS = """
:root{--surface:#fcfcfb;--panel:#ffffff;--text:#0b0b0b;--text2:#52514e;--line:#e4e3df;
--accent:#2a78d6;--good:#0ca30c;--bad:#d03b3b;--warn:#b07a00}
*{box-sizing:border-box}
body{margin:0;background:var(--surface);color:var(--text);font:15px/1.55 -apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
header{background:#0d366b;color:#fff;padding:36px 16px}
header .wrap{max-width:1120px;margin:0 auto}
header h1{margin:0 0 6px;font-size:28px;line-height:1.2}
header p{margin:0;color:#cde2fb}
main{max-width:1120px;margin:0 auto;padding:24px 16px 64px}
nav{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 18px;margin:0 0 28px}
nav ol{margin:0;padding-left:28px;columns:2;column-gap:40px}
nav a{color:var(--accent);text-decoration:none}
section{margin:0 0 40px}
h2{font-size:21px;margin:0 0 4px;padding-top:8px;border-top:2px solid var(--text)}
h3{font-size:16px;margin:22px 0 6px}
.lead{color:var(--text2);margin:4px 0 14px}
.kpis{display:flex;flex-wrap:wrap;gap:12px;margin:14px 0}
.kpis .kpi{flex:1 1 220px}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.kpi-label{font-size:12px;color:var(--text2);text-transform:uppercase;letter-spacing:.04em}
.kpi-value{font-size:26px;font-weight:700;margin:4px 0;font-variant-numeric:tabular-nums}
.kpi-note{font-size:12px;color:var(--text2)}
.kpi.good .kpi-value{color:var(--good)} .kpi.bad .kpi-value{color:var(--bad)}
figure{margin:14px 0;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px}
figure img{max-width:100%;height:auto;display:block;margin:0 auto}
figcaption{font-size:12.5px;color:var(--text2);margin-top:6px}
.grid2{display:flex;flex-wrap:wrap;gap:12px}
.grid2>*{flex:1 1 360px;min-width:0}
@media(max-width:760px){nav ol{columns:1}}
.tblwrap{overflow-x:auto;background:var(--panel);border:1px solid var(--line);border-radius:10px;margin:10px 0}
table.tbl{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
table.tbl th{background:#f3f2ee;text-align:left;font-weight:600;padding:7px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
table.tbl td{padding:6px 10px;border-bottom:1px solid var(--line);vertical-align:top}
table.tbl tr:last-child td{border-bottom:none}
table.tbl td:first-child{white-space:nowrap}
.callout{background:#eef5fd;border-left:4px solid var(--accent);padding:12px 16px;border-radius:6px;margin:12px 0}
.synth{background:#fff6e0;border-left:4px solid #eda100;padding:10px 14px;border-radius:6px;margin:12px 0;font-size:13.5px}
ul.findings li{margin:4px 0}
footer{color:var(--text2);font-size:12px;border-top:1px solid var(--line);padding-top:12px}
"""


def _tw(inner):
    return f'<div class="tblwrap">{inner}</div>'


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
def build_html(R: dict) -> str:
    T, F = R["tables"], R["figures"]
    kpi, bt, comp = T["kpi"], T["backtest_summary"], T["model_comparison"]
    ov, rs, meta = T["outreach_overall"], T["risk_summary"], T["meta"]
    as_of = pd.Timestamp(meta["as_of"])
    m_name, s_name = POLICY_NAMES["model"], POLICY_NAMES["static"]
    auc_m = comp.loc["LightGBM + calibration (target model)", "ROC_AUC"]
    auc_h = comp.loc["Segment heuristic (current practice)", "ROC_AUC"]
    eda = T["eda_summary"]

    sections = []

    # 1. Executive summary ------------------------------------------------------
    tiles = "".join([
        kpi_tile("Net benefit per year", fmt_eur(kpi["Net benefit per year [EUR]"]),
                 "2 hotels, after running costs", "good"),
        kpi_tile("First-year ROI", fmt_pct(kpi["First-year ROI"], 0),
                 f"project cost {fmt_eur(kpi['Project cost [EUR]'])}"),
        kpi_tile("Payback period", f"{_num(kpi['Payback period [months]'], 1)} months", ""),
        kpi_tile("Backtest 2017: ML model", fmt_eur(bt.loc[m_name, "Net result [EUR]"]),
                 f"{fmt_pct(bt.loc[m_name, '% of potential'], 0)} of the maximum potential", "good"),
        kpi_tile("Backtest 2017: static overbooking", fmt_eur(bt.loc[s_name, "Net result [EUR]"]),
                 f"{_num(bt.loc[s_name, 'Guests walked'])} guests walked", "bad"),
        kpi_tile("Model accuracy (ROC AUC)", _num(auc_m, 3),
                 f"segment heuristic: {_num(auc_h, 3)}"),
    ])
    late_rev = eda["Revenue cancelled late (<=7 days) [EUR]"].sum()
    findings = [
        f"In the hotels analysed, {fmt_pct(eda.loc['City hotel', 'Cancellation rate'], 0)} of bookings at the city hotel "
        f"and {fmt_pct(eda.loc['Resort hotel', 'Cancellation rate'], 0)} at the resort were cancelled; "
        f"cancellations in the final week before arrival covered bookings worth {fmt_eur(late_rev)}.",
        f"The risk model (LightGBM with calibration) predicts no-arrival for each individual booking "
        f"with a ROC AUC of {_num(auc_m, 3)} versus {_num(auc_h, 3)} for the segment heuristic, and its probabilities are calibrated.",
        f"In the 2017 backtest (decision at D-{C.DECISION_HORIZON_DAYS}) the model's recommendations delivered "
        f"{fmt_eur(bt.loc[m_name, 'Net result [EUR]'])} net with {_num(bt.loc[m_name, 'Guests walked'])} guests walked, "
        f"while static {int(C.STATIC_OVERBOOKING_PCT * 100)}% overbooking produced {fmt_eur(bt.loc[s_name, 'Net result [EUR]'])}.",
        f"Contacting guests with a risk above {fmt_pct(ov[THRESHOLD_KEY], 0)} "
        f"(about {fmt_pct(ov['Share of bookings to contact'], 0)} of bookings) pays off in the simulation (synthetic data).",
        f"As of {as_of:%d %b %Y}: {rs['Phone contact']} bookings to contact by phone and "
        f"{rs['E-mail contact']} by e-mail over the next 14 days.",
    ]
    sections.append(("summary", "Executive summary",
                     f'<div class="kpis">{tiles}</div><ul class="findings">'
                     + "".join(f"<li>{html.escape(f)}</li>" for f in findings) + "</ul>"
                     + '<div class="synth">The financial results partly rely on synthetic assumptions '
                       '(cost of walking a guest, turned-away demand on sold-out nights, effect of guest contact). '
                       'The full list is in the "Assumptions and limitations" section.</div>'))

    # 2. Diagnosis ----------------------------------------------------------------
    sections.append(("diagnosis", "Diagnosis: scale of the cancellation problem",
                     '<p class="lead">Data: the public Hotel Booking Demand dataset (city hotel and resort, '
                     'arrivals 07.2015-08.2017).</p>'
                     + _tw(table_html(eda, decimals={"Bookings": 0, "Avg. booking lead time (days)": 0},
                                     transpose=True))
                     + fig_html(F["eda_trend"]) + '<div class="grid2">' + fig_html(F["eda_lead"])
                     + fig_html(F["eda_segment"]) + "</div>" + fig_html(F["eda_timing"])
                     + fig_html(F["eda_occupancy"], "Capacity estimated as the maximum realised occupancy.")))

    # 3. Model --------------------------------------------------------------------
    comp_t = comp.copy()
    comp_t.index.name = "Model"
    sections.append(("model", "No-arrival risk model",
                     '<p class="lead">The model scores risk on a specific day before arrival, using only '
                     'information available on that day (leakage-free snapshots). Time-based validation: '
                     'training 2015/16, calibration H2 2016, test 2017.</p>'
                     + _tw(table_html(comp_t, decimals={"ROC_AUC": 3, "PR_AUC": 3, "Brier": 4, "LogLoss": 4}))
                     + '<div class="grid2">' + fig_html(F["model_roc"]) + fig_html(F["model_calibration"]) + "</div>"
                     + fig_html(F["model_auc_horizon"])
                     + _tw(table_html(T["model_by_horizon"], decimals={"ROC AUC (heuristic)": 3, "ROC AUC (LightGBM)": 3,
                                                                        "Brier (heuristic)": 4, "Brier (LightGBM)": 4}))))

    # 4. SHAP -----------------------------------------------------------------------
    sections.append(("drivers", "Why guests do not arrive",
                     fig_html(F["shap_importance"]) + fig_html(F["shap_categories"]) + fig_html(F["shap_dependence"])
                     + '<div class="callout">Price (ADR) and arrival month were deliberately excluded from the model: '
                       'with one year of history the model memorised the season of a specific year, and prices drift '
                       'over time, which hurt the results on 2017 data.</div>'))

    # 5. Occupancy forecast -----------------------------------------------------------
    sections.append(("forecast", "Occupancy forecast",
                     '<p class="lead">The structural forecast (bookings x risk + expected new bookings) '
                     'is corrected by a stage-2 model that learns the hotel\'s sales practice. '
                     'Uncertainty comes from cross-validation residuals (conformal approach).</p>'
                     + fig_html(F["forecast_accuracy"]) + _tw(table_html(T["forecast_accuracy"],
                                                                         decimals={"Nights": 0}))))

    # 6. Backtest -----------------------------------------------------------------
    sections.append(("backtest", "Overbooking: 2017 backtest",
                     f'<p class="lead">For every night in 2017 the decision was taken at D-{C.DECISION_HORIZON_DAYS}, '
                     'based solely on what was known at the time. The result is the revenue from rooms that would '
                     'otherwise have stayed empty, minus the cost of walked guests.</p>'
                     + fig_html(F["ob_example"]) + fig_html(F["ob_cumulative"]) + fig_html(F["ob_bars"])
                     + _tw(table_html(bt))
                     + '<div class="synth">Turned-away demand on sold-out nights and the walk cost are synthetic '
                       '(the dataset only contains accepted bookings).</div>'))

    # 7. Plan -------------------------------------------------------------------
    plan = T["forward_plan"].copy()
    plan_t = plan.assign(Hotel=plan["hotel"].map(HOTEL_NAMES))[[
        "Hotel", "night", "horizon", "capacity", "in_house", "on_books", "forecast", "occ_p10", "occ_p90",
        "p_full", "x_opt", "exp_gain", "recommendation"]].rename(columns={
            "night": "Night", "horizon": "Days to night", "capacity": "Capacity", "in_house": "Guests in house",
            "on_books": "Bookings on the books", "forecast": "Occupancy forecast", "occ_p10": "P10",
            "occ_p90": "P90", "p_full": "Probability of full hotel", "x_opt": "Extra bookings",
            "exp_gain": "Expected gain [EUR]", "recommendation": "Recommendation"})
    plan_t = plan_t.sort_values(["Night", "Hotel"]).head(28)
    sections.append(("plan", f"Overbooking plan from {as_of:%d %b %Y}",
                     fig_html(F["forward_plan"]) + "<h3>Next 14 nights</h3>"
                     + _tw(table_html(plan_t, index=False, decimals={"Occupancy forecast": 0}))
                     + '<p class="lead">Full 60-day plan: the "Overbooking_plan" sheet in the Excel file.</p>'))

    # 8. Risk list --------------------------------------------------------------
    rl = T["risk_list"].drop(columns=["Did not arrive (verification)"])
    rs_t = pd.DataFrame([rs], index=["Value"])
    sections.append(("risk", "High-risk bookings (next 14 days)",
                     _tw(table_html(rs_t, index=False))
                     + _tw(table_html(rl, index=False, max_rows=25, decimals={"Days to arrival": 0, "Nights": 0,
                                                                              "Booking ID (pseudonym)": 0}))
                     + '<p class="lead">Full list: the "Risk_list" sheet. IDs are pseudonyms - '
                       'the model does not need guests\' personal data.</p>'))

    # 9. Preventive actions -----------------------------------------------------
    ov_t = pd.DataFrame([ov], index=["Value"])
    sections.append(("prevention", "Preventive actions: A/B test of guest contact",
                     '<div class="synth">This section is a SIMULATION: the effect of contact was applied synthetically '
                     'to real bookings to demonstrate the evaluation method and how to choose the target group. '
                     'After go-live it is replaced by the result of a real test.</div>'
                     + _tw(table_html(ov_t, index=False)) + fig_html(F["outreach_uplift"])
                     + _tw(table_html(T["outreach_deciles"]))))

    # 10. Policies --------------------------------------------------------------
    wi = T["policy_whatif"]
    sections.append(("policies", "Cancellation and prepayment policies",
                     fig_html(F["policy_heat"]) + fig_html(F["policy_whatif"])
                     + _tw(table_html(wi[["Hotel", "Channel", "Lead time (days)", "Bookings", "Cancellation rate",
                                          "Late cancellation rate", "Revenue change per year [EUR]",
                                          "Recommendation"]], index=False, max_rows=12))
                     + _tw(table_html(T["channels"], index=False))))

    # 11. Monitoring ------------------------------------------------------------
    sections.append(("monitoring", "Model monitoring",
                     '<div class="grid2">' + fig_html(F["drift"]) + _tw(table_html(T["drift"], index=False, max_rows=14,
                                                                                    decimals={"PSI": 3})) + "</div>"
                     + fig_html(F["monthly"])
                     + _tw(table_html(T["monthly"], decimals={"ROC AUC": 3, "Brier": 4}))
                     + '<p class="lead">Rule: PSI &gt; 0.25 or calibration error &gt; 3 pp for 2 months '
                       '→ retrain and recalibrate.</p>'))

    # 12. Business case -----------------------------------------------------------
    items = T["business_items"]
    sections.append(("business", "Business case",
                     fig_html(F["business"]) + _tw(table_html(items))
                     + '<p class="lead">The backtest result (8 months) was annualised in proportion to the number of nights. '
                       'Across a hotel chain the effect scales with the number of properties at a near-zero marginal cost '
                       '(same code and a central model).</p>'))

    # 13. Assumptions --------------------------------------------------------------
    limits = [
        "The dataset covers 2 hotels in Portugal (2015-2017); results must be confirmed on the client's data.",
        "The room count is not in the data - the maximum realised occupancy was used.",
        "The dataset only contains accepted bookings; turned-away demand on sold-out nights is synthetic.",
        "Analysis is at hotel level (not room type); extending to room types needs data on room-type capacity and upgrades.",
        "The 'country' variable and fields filled in after the fact were excluded (information leakage).",
        "Annualising the backtest result (January-August) may overstate the effect because it omits the low autumn season.",
    ]
    sections.append(("assumptions", "Assumptions and limitations",
                     _tw(table_html(T["assumptions"])) + "<ul>" + "".join(f"<li>{html.escape(x)}</li>" for x in limits)
                     + "</ul>"))

    nav = "".join(f'<li><a href="#{sid}">{html.escape(title)}</a></li>' for sid, title, _ in sections)
    body = "".join(f'<section id="{sid}"><h2>{html.escape(title)}</h2>{content}</section>'
                   for sid, title, content in sections)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cancellation and overbooking report</title><style>{CSS}</style></head><body>
<header><div class="wrap"><h1>Cancellation forecasting and smart overbooking</h1>
<p>Client report · data: Hotel Booking Demand · as of {as_of:%d %b %Y} ·
generated {datetime.now():%d %b %Y %H:%M}</p></div></header>
<main><nav><strong>Contents</strong><ol>{nav}</ol></nav>{body}
<footer>Data source: Antonio, Almeida, Nunes (2019), "Hotel booking demand datasets", Data in Brief 22, CC BY 4.0 licence.
Items marked SYNTHETIC are assumptions to be replaced with the client's data.</footer></main></body></html>"""


def write_html(R: dict, path=None):
    path = path or C.REPORTS / "client_report.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_html(R), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Excel
# ---------------------------------------------------------------------------
def write_excel(R: dict, path=None):
    path = path or C.REPORTS / "project_results.xlsx"
    T = R["tables"]
    plan = T["forward_plan"].copy()
    plan["hotel"] = plan["hotel"].map(HOTEL_NAMES)
    kpi = pd.DataFrame({"Value": T["kpi"]})
    sheets = {
        "Summary": kpi,
        "Overbooking_plan": plan,
        "Risk_list": T["risk_list"],
        "Backtest_summary": T["backtest_summary"],
        "Backtest_nights": T["backtest_nights"],
        "Model_comparison": T["model_comparison"],
        "Model_by_horizon": T["model_by_horizon"],
        "Feature_importance": T["feature_importance"],
        "Forecast_accuracy": T["forecast_accuracy"],
        "Prevention_deciles": T["outreach_deciles"],
        "Policy_whatif": T["policy_whatif"],
        "Channels": T["channels"],
        "Monitoring_PSI": T["drift"],
        "Monitoring_monthly": T["monthly"],
        "Business_case": T["business_items"],
        "Assumptions": T["assumptions"],
        "Diagnosis": T["eda_summary"],
    }
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, d in sheets.items():
            d = d.copy()
            keep_index = not isinstance(d.index, pd.RangeIndex)
            d.to_excel(xw, sheet_name=name[:31], index=keep_index)
            ws = xw.sheets[name[:31]]
            ws.freeze_panes = "B2" if keep_index else "A2"
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col[:200])
                ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 60)
    return path


# ---------------------------------------------------------------------------
# Display in Jupyter / Spyder / PyCharm
# ---------------------------------------------------------------------------
def show(obj, decimals: dict | None = None, max_rows: int | None = 30, transpose: bool = False):
    """Displays a chart or a (formatted) table in a notebook / IPython console;
    in a plain Python console it prints the table as text and opens the chart in a window."""
    import matplotlib.figure
    try:
        from IPython.display import display
        ipy = True
    except ImportError:  # pragma: no cover
        ipy = False
    if isinstance(obj, matplotlib.figure.Figure):
        if ipy:
            display(obj)
        else:
            obj.show()
        return
    if isinstance(obj, dict):                     # KPI dictionary -> vertical table
        obj = pd.DataFrame([obj], index=["Value"])
        transpose = True
    if isinstance(obj, pd.DataFrame):
        d = obj.head(max_rows) if max_rows else obj
        f = format_table(d, decimals)
        if transpose:
            f = f.T
        if ipy:
            display(f)
        else:
            print(f.to_string())
