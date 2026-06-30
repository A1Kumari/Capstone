"""
ui/streamlit_app.py — DischargeAI Clinical Auditor (single-page, tabbed)
"""

import sys
import uuid
import tempfile
import os
import json
import logging
from pathlib import Path

import streamlit as st

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from langgraph.types import Command
from agents.extractor_agent import extract_clinical_data
from langgraph_workflows.clinical_auditor_workflow import discharge_pipeline

log = logging.getLogger("StreamlitApp")


# ─────────────────────────────────────────────────────────────
#  Log capture
# ─────────────────────────────────────────────────────────────
class _LogCapture(logging.Handler):
    def emit(self, record):
        if "pipeline_logs" in st.session_state:
            try:
                st.session_state["pipeline_logs"].append(self.format(record))
            except Exception:
                pass

_log_handler = _LogCapture()
_log_handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s — %(message)s"))

def _attach(): st.session_state["pipeline_logs"] = []; logging.getLogger().addHandler(_log_handler)
def _detach(): logging.getLogger().removeHandler(_log_handler)

def _thread_config():
    return {"configurable": {"thread_id": st.session_state["thread_id"]}}

def _graph_state():
    return discharge_pipeline.get_state(_thread_config())

def _is_interrupted():
    try:
        return bool(_graph_state().next)
    except Exception:
        return False

def _interrupt_payload():
    try:
        gs = _graph_state()
        interrupts = [intr for task in gs.tasks for intr in task.interrupts]
        return interrupts[0].value if interrupts else {}
    except Exception:
        return {}

def _switch_tab(index: int):
    """Inject JS to programmatically click tab at the given 0-based index."""
    st.markdown(
        f'<script>setTimeout(function(){{'
        f'var tabs=window.parent.document.querySelectorAll(\'[data-baseweb="tab"]\');'
        f'if(tabs[{index}])tabs[{index}].click();'
        f'}},150);</script>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
#  Page config & design system
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DischargeAI",
    page_icon=":material/clinical_notes:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root{
  --paper:        #FAF9F6;
  --paper-raised: #FFFFFF;
  --ink:          #1B1D1F;
  --ink-soft:     #5B6166;
  --ink-faint:    #8A9097;
  --line:         #E3E0D8;
  --accent:       #0B6E6E;
  --accent-dark:  #095858;
  --accent-soft:  #E4F1F0;
  --sidebar:      #14181B;
  --sidebar-line: #262C30;
  --sidebar-text: #C9CFD3;
  --ok:      #1F7A4D; --ok-bg:#E8F5EC;  --ok-line:#BFE3CC;
  --warn:    #9A6B14; --warn-bg:#FBF1DD; --warn-line:#EFD6A0;
  --danger:  #A6334A; --danger-bg:#FBE9ED; --danger-line:#EFC0CB;
  --info:    #2A5C8A; --info-bg:#E8F0F8; --info-line:#BFD7EC;
  --neutral: #5B6166; --neutral-bg:#F0EEE9; --neutral-line:#D9D5CC;
}

/* ===== layout — pull content to top ===== */
[data-testid="stHeader"]{ display:none !important; }
[data-testid="stDecoration"]{ display:none !important; }
.block-container{ padding-top:1.5rem !important; padding-bottom:3rem !important; }
section[data-testid="stSidebar"] > div:first-child{ padding-top:1.5rem !important; }

/* ===== base typography ===== */
html, body, [data-testid="stAppViewContainer"]{
  background: var(--paper);
  font-family: 'IBM Plex Sans', sans-serif;
  color: var(--ink);
}
h1,h2,h3,h4{
  font-family:'IBM Plex Serif', serif;
  font-weight:600;
  color: var(--ink);
  letter-spacing:-0.01em;
}

/* ===== sidebar ===== */
[data-testid="stSidebar"]{
  background:var(--sidebar);
  border-right:1px solid var(--sidebar-line);
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] .stMarkdown{ color:var(--sidebar-text) !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3{ color:#F3F1EC !important; }
[data-testid="stSidebar"] hr{ border-color:var(--sidebar-line); }

/* badges inside sidebar always get dark text so they're readable on their light bg */
[data-testid="stSidebar"] .badge,
[data-testid="stSidebar"] span[class*="badge"]{ color:#1B1D1F !important; }

/* sidebar buttons */
[data-testid="stSidebar"] .stButton > button,
[data-testid="stSidebar"] [data-testid^="stBaseButton"]{
  background: rgba(255,255,255,0.07) !important;
  color: #E2E8F0 !important;
  border: 1px solid rgba(255,255,255,0.15) !important;
  font-family: 'IBM Plex Sans', sans-serif !important;
  font-size: .85rem !important;
  border-radius: 5px !important;
  width: 100%;
}
[data-testid="stSidebar"] .stButton > button:hover,
[data-testid="stSidebar"] [data-testid^="stBaseButton"]:hover{
  background: var(--accent) !important;
  color: #ffffff !important;
  border-color: var(--accent) !important;
}

/* ===== eyebrow / section labels ===== */
.eyebrow{
  font-family:'IBM Plex Mono', monospace;
  font-size:.68rem; letter-spacing:.14em; text-transform:uppercase;
  color: var(--accent); font-weight:600; margin-bottom:2px;
}
.section-title{ margin-top:0; margin-bottom:4px; }
.section-sub{ color:var(--ink-soft); font-size:.88rem; margin-bottom:16px; }

/* ===== status dot + badge ===== */
.dot{ display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; vertical-align:middle; }
.dot-ok{ background:var(--ok); } .dot-warn{ background:var(--warn); }
.dot-danger{ background:var(--danger); } .dot-info{ background:var(--info); }
.dot-neutral{ background:var(--ink-faint); }

.badge{
  display:inline-flex; align-items:center; padding:3px 10px; border-radius:3px;
  font-size:.7rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase;
  font-family:'IBM Plex Mono', monospace; border:1px solid;
}
.badge-ok      { background:var(--ok-bg);      color:var(--ok);      border-color:var(--ok-line); }
.badge-warn    { background:var(--warn-bg);    color:var(--warn);    border-color:var(--warn-line); }
.badge-danger  { background:var(--danger-bg);  color:var(--danger);  border-color:var(--danger-line); }
.badge-info    { background:var(--info-bg);    color:var(--info);    border-color:var(--info-line); }
.badge-neutral { background:var(--neutral-bg); color:var(--neutral); border-color:var(--neutral-line); }

/* ===== cards ===== */
.card{
  background:var(--paper-raised); border:1px solid var(--line);
  border-radius:8px; padding:16px 20px; margin-bottom:12px;
}
.card-tight{ padding:10px 14px; }

.field-row{
  display:flex; align-items:center; gap:10px; padding:7px 0;
  border-bottom:1px solid #F0EEE9; font-size:.86rem;
}
.field-row:last-child{ border-bottom:none; }
.field-label{ min-width:175px; color:var(--ink-soft); font-size:.84rem; }
.field-value{ color:var(--ink); font-family:'IBM Plex Mono', monospace; font-size:.81rem; }
.field-missing{ color:var(--danger); font-style:italic; }

/* ===== risk panel ===== */
.risk-panel{ border-radius:8px; padding:18px 22px; border-left:4px solid; margin-bottom:16px; }
.risk-low    { background:var(--ok-bg);     border-color:var(--ok); }
.risk-medium { background:var(--warn-bg);   border-color:var(--warn); }
.risk-high   { background:var(--danger-bg); border-color:var(--danger); }
.risk-label  { font-size:1.1rem; font-weight:700; font-family:'IBM Plex Serif', serif; }
.risk-meta   { color:var(--ink-soft); font-size:.88rem; margin-top:4px; }

/* ===== notice boxes ===== */
.notice{
  border-radius:6px; border:1px solid; border-left-width:4px;
  padding:14px 18px; margin:12px 0;
}
.notice-warn{ background:var(--warn-bg); border-color:var(--warn-line); border-left-color:var(--warn); }
.notice-info{ background:var(--info-bg); border-color:var(--info-line); border-left-color:var(--info); }
.notice strong{ color:var(--ink); }
.notice span{ color:var(--ink-soft); font-size:.88rem; }

/* ===== divider ===== */
.rule{ border:none; border-top:1px solid var(--line); margin:24px 0; }

/* ===== top-level tabs ===== */
[data-baseweb="tab-list"]{
  gap:2px; border-bottom:2px solid var(--line); background:transparent;
}
[data-baseweb="tab"]{
  font-family:'IBM Plex Sans', sans-serif; font-size:.82rem; font-weight:500;
  color:var(--ink-soft); padding:10px 20px;
  background:transparent; border:none; border-bottom:2px solid transparent;
  margin-bottom:-2px; border-radius:0;
}
[aria-selected="true"][data-baseweb="tab"]{
  color:var(--accent); font-weight:600; border-bottom-color:var(--accent);
}
[data-baseweb="tab-highlight"]{ display:none; }
[data-baseweb="tab-border"]{ display:none; }

/* ===== primary action button ===== */
.stButton > button,
[data-testid^="stBaseButton"]{
  border-radius:5px !important;
  font-family:'IBM Plex Sans', sans-serif !important;
  font-weight:500 !important;
  transition: all .15s ease !important;
}
.stButton > button[kind="primary"],
[data-testid="stBaseButton-primary"]{
  background: var(--accent) !important;
  color: #ffffff !important;
  border-color: var(--accent) !important;
}
.stButton > button[kind="primary"]:hover,
[data-testid="stBaseButton-primary"]:hover{
  background: var(--accent-dark) !important;
  border-color: var(--accent-dark) !important;
  color: #ffffff !important;
}

/* ===== download buttons ===== */
.stDownloadButton > button,
[data-testid="stDownloadButton"] button,
[data-testid="stBaseButton-download"]{
  background: var(--paper-raised) !important;
  color: var(--accent) !important;
  border: 1.5px solid var(--accent) !important;
  border-radius: 5px !important;
  font-family: 'IBM Plex Sans', sans-serif !important;
  font-weight: 500 !important;
}
.stDownloadButton > button:hover,
[data-testid="stDownloadButton"] button:hover,
[data-testid="stBaseButton-download"]:hover{
  background: var(--accent-soft) !important;
  color: var(--accent-dark) !important;
}

/* ===== file upload ===== */
.upload-chip{ font-family:'IBM Plex Mono', monospace; font-size:.7rem; }
[data-testid="stFileUploader"]{ padding:4px 0; }

/* ===== metrics ===== */
[data-testid="stMetricValue"]{ color:var(--ink); font-family:'IBM Plex Serif', serif; }
[data-testid="stMetricLabel"]{ color:var(--ink-soft); font-size:.8rem !important; text-transform:uppercase; letter-spacing:.06em; }
</style>
""", unsafe_allow_html=True)


def section_header(eyebrow: str, title: str, sub: str = ""):
    st.markdown(f'<div class="eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.markdown(f'<h3 class="section-title">{title}</h3>', unsafe_allow_html=True)
    if sub:
        st.markdown(f'<div class="section-sub">{sub}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  Session state
# ─────────────────────────────────────────────────────────────
DEFAULTS = {
    "stage"           : "upload",
    "thread_id"       : str(uuid.uuid4()),
    "extracted"       : None,
    "translation"     : None,
    "pipeline_state"  : None,
    "hitl_payload"    : {},
    "pipeline_logs"   : [],
    "qa_history"      : [],
    "qa_patient_scope": None,   # tracks which patient the chat history belongs to
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

def _reset():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.session_state["thread_id"] = str(uuid.uuid4())  # new thread = fresh memory


FIELD_LABELS = {
    "patient_id":"Patient ID","patient_name":"Patient Name","age":"Age",
    "gender":"Gender","address":"Address","admission_date":"Admission Date",
    "discharge_date":"Discharge Date","ward":"Ward","bed_no":"Bed No.",
    "attending_physician":"Attending Physician","consulting_doctors":"Consulting Doctors",
    "discharge_diagnosis":"Discharge Diagnosis","allergies":"Allergies",
    "follow_up_appointment":"Follow-up Appointment","discharge_instructions":"Discharge Instructions",
}
ALL_PATIENT_FIELDS = list(FIELD_LABELS.keys())
PRESCRIPTION_FIELDS = ["sl_no","medicine_name","strength","dosage",
                       "frequency","route","period","remarks","total_quantity"]


# ─────────────────────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────────────────────
stage = st.session_state["stage"]
ps    = st.session_state.get("pipeline_state")

with st.sidebar:
    st.markdown(
        '<div style="margin-bottom:4px">'
        '<span style="font-family:IBM Plex Serif,serif;font-size:1.25rem;'
        'font-weight:700;color:#F3F1EC !important">DischargeAI</span>'
        '</div>'
        '<div style="font-size:.75rem;color:#8A9097;letter-spacing:.06em;'
        'text-transform:uppercase;margin-bottom:16px">Clinical Discharge Auditor</div>',
        unsafe_allow_html=True,
    )

    STAGE_BADGE = {
        "upload": ("neutral", "Ready"),
        "hitl"  : ("warn",    "Review required"),
        "done"  : ("ok",      "Complete"),
        "failed": ("danger",  "Failed"),
    }
    kind, label = STAGE_BADGE.get(stage, ("neutral", "—"))
    st.markdown(
        f'<div style="margin-bottom:8px;font-size:.7rem;color:#8A9097;'
        f'text-transform:uppercase;letter-spacing:.08em">Status</div>'
        f'<span class="dot dot-{kind}"></span>'
        f'<span class="badge badge-{kind}" style="font-size:.68rem">{label}</span>',
        unsafe_allow_html=True,
    )

    if ps and ps.get("risk_level"):
        risk = ps["risk_level"]
        rc   = {"low":"ok","medium":"warn","high":"danger"}.get(risk,"neutral")
        st.markdown(
            f'<div style="margin-top:10px;font-size:.7rem;color:#8A9097;'
            f'text-transform:uppercase;letter-spacing:.08em">Risk Level</div>'
            f'<span class="badge badge-{rc}" style="margin-top:4px;font-size:.68rem">'
            f'{risk.upper()}</span>',
            unsafe_allow_html=True,
        )

    st.markdown('<hr style="border-color:#262C30;margin:16px 0">', unsafe_allow_html=True)

    if st.button("New case", use_container_width=True):
        _reset()
        st.rerun()

    st.markdown('<hr style="border-color:#262C30;margin:16px 0">', unsafe_allow_html=True)

    _stage_now = st.session_state.get("stage", "idle")
    STEPS = [
        ("Extraction",   bool(st.session_state.get("extracted")),              None),
        ("Translation",  bool(st.session_state.get("translation")),            None),
        ("Completeness", bool(ps and ps.get("completeness")),                  None),
        ("Human review", _stage_now == "done" or bool(ps and ps.get("report")), _stage_now == "hitl"),
        ("EHR validation", bool(ps and ps.get("ehr_validation")),              None),
        ("Audit report", bool(ps and ps.get("report")),                        None),
    ]
    st.markdown(
        '<div style="font-size:.7rem;color:#8A9097;text-transform:uppercase;'
        'letter-spacing:.08em;margin-bottom:10px">Pipeline</div>',
        unsafe_allow_html=True,
    )
    for step_name, done, pending in STEPS:
        if pending:
            dot_cls = "dot-warn"
            col = "var(--warn)"
        elif done:
            dot_cls = "dot-ok"
            col = "#C9CFD3"
        else:
            dot_cls = "dot-neutral"
            col = "#5B6166"
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;padding:4px 0;'
            f'font-size:.82rem;color:{col}">'
            f'<span class="dot {dot_cls}" style="flex-shrink:0"></span>'
            f'{step_name}</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────
#  Header
# ─────────────────────────────────────────────────────────────
st.markdown(
    '<div style="display:flex;align-items:baseline;gap:14px;margin-bottom:2px">'
    '<div class="eyebrow" style="margin:0">Clinical Auditor</div>'
    '</div>'
    '<h1 style="margin:0 0 4px 0;font-size:1.9rem">DischargeAI</h1>'
    '<div class="section-sub" style="margin-bottom:12px">Upload a discharge report, '
    'lab report, and hospital bill to run the full audit pipeline.</div>',
    unsafe_allow_html=True,
)

from fpdf import FPDF

def generate_pdf_report(report: dict, normalized: dict | None = None) -> bytes:
    nm  = normalized or {}
    pdf = FPDF()
    pdf.set_margins(15, 15, 15)
    pdf.add_page()
    W = pdf.w - pdf.l_margin - pdf.r_margin

    def s(v) -> str:
        return str(v if v not in (None, "", [], {}) else "—").encode("latin-1", "replace").decode("latin-1")

    def body(text: str, line_h: int = 6):
        pdf.set_font("helvetica", size=10)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(W, line_h, text=s(text))

    def section(title: str):
        pdf.ln(5)
        pdf.set_fill_color(15, 107, 107)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("helvetica", "B", 10)
        pdf.set_x(pdf.l_margin)
        pdf.cell(W, 8, text=s(title), new_x="LMARGIN", new_y="NEXT", fill=True)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("helvetica", size=10)
        pdf.ln(2)

    def row(label: str, value, col_w: float = 45):
        pdf.set_font("helvetica", "B", 9)
        pdf.set_x(pdf.l_margin)
        pdf.cell(col_w, 6, text=s(label))
        pdf.set_font("helvetica", size=9)
        pdf.multi_cell(W - col_w, 6, text=s(value))

    def divider():
        pdf.set_draw_color(200, 200, 200)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + W, pdf.get_y())
        pdf.ln(2)

    # ── Cover header ──────────────────────────────────────────
    pdf.set_fill_color(15, 107, 107)
    pdf.rect(0, 0, pdf.w, 28, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("helvetica", "B", 17)
    pdf.set_xy(pdf.l_margin, 8)
    pdf.cell(W, 9, text="CLINICAL DISCHARGE AUDIT REPORT", align="C",
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", size=9)
    pdf.set_xy(pdf.l_margin, 19)
    pdf.cell(W, 5,
             text=s(f"Generated: {report.get('generated_at','—')}   |   Rules: {report.get('rules_version','—')[:12]}…"),
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)

    # ── Risk banner ───────────────────────────────────────────
    risk  = (report.get("risk_level") or "unknown").upper()
    score = report.get("risk_score", 0)
    rec   = report.get("recommendation", "—")
    fill  = {"LOW": (220, 252, 231), "MEDIUM": (254, 243, 199), "HIGH": (254, 226, 226)}.get(risk, (240,240,240))
    text  = {"LOW": (21, 128, 61),   "MEDIUM": (146, 64, 14),   "HIGH": (185, 28, 28)}.get(risk, (60,60,60))
    pdf.set_fill_color(*fill)
    pdf.set_text_color(*text)
    pdf.set_font("helvetica", "B", 13)
    pdf.set_x(pdf.l_margin)
    pdf.cell(W, 12, text=f"RISK LEVEL: {risk}   |   Score: {score}", fill=True,
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", size=9)
    pdf.set_x(pdf.l_margin)
    pdf.cell(W, 6, text=f"Recommendation: {s(rec)}", fill=True, align="C",
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    # ── Patient demographics ──────────────────────────────────
    section("PATIENT INFORMATION")
    col1_fields = [
        ("Patient Name",   nm.get("patient_name")),
        ("Patient ID",     nm.get("patient_id")),
        ("Age",            nm.get("age")),
        ("Gender",         nm.get("gender")),
        ("Address",        nm.get("address")),
    ]
    col2_fields = [
        ("Admission Date", nm.get("admission_date")),
        ("Discharge Date", nm.get("discharge_date")),
        ("Ward / Bed",     f"{nm.get('ward','—')} / {nm.get('bed_no','—')}"),
        ("Attending Dr.",  nm.get("attending_physician")),
        ("Service Line",   nm.get("service_line")),
    ]
    half = W / 2 - 2
    y_start = pdf.get_y()
    x_left  = pdf.l_margin
    x_right = pdf.l_margin + half + 4

    for label, val in col1_fields:
        pdf.set_xy(x_left, pdf.get_y())
        pdf.set_font("helvetica", "B", 8); pdf.cell(38, 5, text=label)
        pdf.set_font("helvetica", size=8); pdf.multi_cell(half - 38, 5, text=s(val))

    y_after_left = pdf.get_y()
    pdf.set_xy(x_right, y_start)

    for label, val in col2_fields:
        pdf.set_xy(x_right, pdf.get_y())
        pdf.set_font("helvetica", "B", 8); pdf.cell(38, 5, text=label)
        pdf.set_font("helvetica", size=8); pdf.multi_cell(half - 38, 5, text=s(val))

    pdf.set_xy(pdf.l_margin, max(y_after_left, pdf.get_y()) + 2)

    # ── Diagnosis ────────────────────────────────────────────
    section("DISCHARGE DIAGNOSIS")
    body(nm.get("discharge_diagnosis"))
    pdf.ln(2)
    row("Allergies",   ", ".join(nm.get("allergies") or []) or "None documented")
    pdf.ln(1)
    row("Follow-up",  nm.get("follow_up_appointment"))
    pdf.ln(2)
    if nm.get("discharge_instructions"):
        pdf.set_font("helvetica", "B", 9); pdf.set_x(pdf.l_margin)
        pdf.cell(W, 5, "Discharge Instructions:", new_x="LMARGIN", new_y="NEXT")
        body(nm.get("discharge_instructions"))
    pdf.ln(1)

    # ── Medications ───────────────────────────────────────────
    section("MEDICATIONS PRESCRIBED")
    meds = nm.get("medications") or []
    if not meds:
        body("No medications recorded.")
    else:
        col_w = [6, 38, 18, 18, 28, 16, 16, W - 140]
        headers = ["#", "Medicine", "Strength", "Dosage", "Frequency", "Route", "Period", "Remarks"]
        pdf.set_fill_color(230, 230, 230)
        pdf.set_font("helvetica", "B", 8)
        pdf.set_x(pdf.l_margin)
        for h, cw in zip(headers, col_w):
            pdf.cell(cw, 6, text=h, border=1, fill=True)
        pdf.ln()
        pdf.set_font("helvetica", size=8)
        for i, med in enumerate(meds, 1):
            pdf.set_x(pdf.l_margin)
            vals = [
                str(i),
                s(med.get("medicine_name")),
                s(med.get("strength")),
                s(med.get("dosage")),
                s(med.get("frequency")),
                s(med.get("route")),
                s(med.get("period")),
                s(med.get("remarks")),
            ]
            fill = i % 2 == 0
            pdf.set_fill_color(248, 248, 248)
            for v, cw in zip(vals, col_w):
                pdf.cell(cw, 6, text=v[:20], border=1, fill=fill)
            pdf.ln()
    pdf.ln(2)

    # ── Lab reports ───────────────────────────────────────────
    section("LAB REPORTS")
    labs = nm.get("lab_results") or []
    row("Vendor", nm.get("lab_vendor"))
    if not labs:
        body("No lab results recorded.")
    else:
        col_w = [38, 28, 16, 38, W - 120]
        headers = ["Test", "Value", "Unit", "Reference Range", "Status"]
        pdf.set_fill_color(230, 230, 230)
        pdf.set_font("helvetica", "B", 8)
        pdf.set_x(pdf.l_margin)
        for h, cw in zip(headers, col_w):
            pdf.cell(cw, 6, text=h, border=1, fill=True)
        pdf.ln()
        pdf.set_font("helvetica", size=8)
        for i, lab in enumerate(labs, 1):
            pdf.set_x(pdf.l_margin)
            vals = [
                s(lab.get("name")), s(lab.get("value")), s(lab.get("unit")),
                s(lab.get("reference_range")), s(lab.get("status")),
            ]
            fill = i % 2 == 0
            pdf.set_fill_color(248, 248, 248)
            for v, cw in zip(vals, col_w):
                pdf.cell(cw, 6, text=v[:22], border=1, fill=fill)
            pdf.ln()
    pdf.ln(2)

    # ── Bill summary ──────────────────────────────────────────
    section("BILLING & RELEASE")
    row("Total Bill",      nm.get("total_bill"))
    row("Payment Status",  "Paid" if nm.get("bill_paid") else "Outstanding")
    row("Discharge Release", "Released" if nm.get("discharge_ok") else "Not approved")
    pdf.ln(2)

    # ── HITL corrections ─────────────────────────────────────
    hitl = report.get("hitl_corrections") or nm.get("_hitl_corrections")
    if hitl:
        section("REVIEWER CORRECTIONS (HITL)")
        for k, v in hitl.items():
            row(k.replace("_", " ").title(), str(v) if not isinstance(v, list) else ", ".join(str(x) for x in v))
        pdf.ln(2)

    # ── Audit flags ───────────────────────────────────────────
    section("AUDIT FLAGS")
    flags = report.get("flags") or []
    if not flags:
        body("No flags raised.")
    else:
        col_rule = 55; col_w_num = 14; col_det = W - col_rule - col_w_num
        pdf.set_fill_color(230, 230, 230)
        pdf.set_font("helvetica", "B", 8)
        pdf.set_x(pdf.l_margin)
        pdf.cell(col_rule, 6, "Rule", border=1, fill=True)
        pdf.cell(col_w_num, 6, "Weight", border=1, fill=True)
        pdf.cell(col_det, 6, "Detail", border=1, fill=True)
        pdf.ln()
        pdf.set_font("helvetica", size=8)
        for i, flag in enumerate(sorted(flags, key=lambda x: -x.get("weight", 0)), 1):
            w   = flag.get("weight", 0)
            fill_col = (254, 226, 226) if w >= 8 else ((254, 243, 199) if w >= 3 else (255, 255, 255))
            pdf.set_fill_color(*fill_col)
            pdf.set_x(pdf.l_margin)
            pdf.cell(col_rule, 6, text=s(flag.get("rule"))[:30], border=1, fill=True)
            pdf.cell(col_w_num, 6, text=f"+{w}", border=1, fill=True, align="C")
            pdf.cell(col_det, 6, text=s(flag.get("detail"))[:70], border=1, fill=True)
            pdf.ln()
    pdf.ln(3)

    # ── Audit trail ───────────────────────────────────────────
    section("AUDIT TRAIL")
    pdf.set_font("helvetica", size=8)
    for entry in (report.get("audit_trail") or []):
        pdf.set_x(pdf.l_margin)
        ts  = entry.get("timestamp", "")
        msg = s(entry.get("message", ""))
        pdf.set_font("helvetica", "B", 8); pdf.cell(38, 5, text=ts)
        pdf.set_font("helvetica", size=8); pdf.multi_cell(W - 38, 5, text=msg)

    # ── Footer on every page ─────────────────────────────────
    class _PDF(type(pdf)): pass   # can't subclass after instantiation, use footer via alias
    return bytes(pdf.output())


# ─────────────────────────────────────────────────────────────
#  Top-level navigation
#  Tab order (0-indexed): Intake=0  Extraction=1  Translation=2
#                         Results=3  Ask=4  Logs=5
# ─────────────────────────────────────────────────────────────
if st.session_state.get("_goto_tab") is not None:
    _switch_tab(st.session_state.pop("_goto_tab"))

tab_intake, tab_extract, tab_translate, tab_results, tab_ask, tab_logs = st.tabs(
    ["Intake", "Extraction", "Translation", "Results", "Ask Records", "Logs"]
)


# ═════════════════════════════════════════════════════════════
#  TAB — Intake (upload & run)
# ═════════════════════════════════════════════════════════════
with tab_intake:
    section_header("Step 01", "Upload documents", "All three files are required before the pipeline can run.")

    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.caption("DISCHARGE REPORT — doctor's summary")
            discharge_file = st.file_uploader("Discharge", label_visibility="collapsed",
                type=["pdf","docx","png","jpg","jpeg","txt","json"], key="up_d")
            if discharge_file:
                st.markdown('<span class="badge badge-ok upload-chip">Ready</span>', unsafe_allow_html=True)
    with c2:
        with st.container(border=True):
            st.caption("LAB REPORT — test results")
            lab_file = st.file_uploader("Lab", label_visibility="collapsed",
                type=["pdf","docx","png","jpg","jpeg","txt","json"], key="up_l")
            if lab_file:
                st.markdown('<span class="badge badge-ok upload-chip">Ready</span>', unsafe_allow_html=True)
    with c3:
        with st.container(border=True):
            st.caption("HOSPITAL BILL — billing summary")
            bill_file = st.file_uploader("Bill", label_visibility="collapsed",
                type=["pdf","docx","png","jpg","jpeg","txt","json"], key="up_b")
            if bill_file:
                st.markdown('<span class="badge badge-ok upload-chip">Ready</span>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    run_ready = all([discharge_file, lab_file, bill_file])
    if not run_ready:
        st.caption(f"{sum(1 for f in [discharge_file,lab_file,bill_file] if f)} of 3 files uploaded")

    if st.button("Run audit pipeline", type="primary", disabled=not run_ready):
        _attach()
        tmp = {}
        for lbl, f in [("discharge",discharge_file),("lab",lab_file),("bill",bill_file)]:
            t = tempfile.NamedTemporaryFile(delete=False, suffix=Path(f.name).suffix)
            t.write(f.read()); t.close()
            tmp[lbl] = t.name

        try:
            with st.status("Running audit pipeline…", expanded=True) as status_box:

                st.write("**Step 1 / 2** — Extracting documents")
                extracted = {}
                for lbl, path in tmp.items():
                    r = extract_clinical_data(path)
                    if r["status"] not in ("success",):
                        st.error(f"Extraction failed for {lbl}: {r.get('error')}")
                        st.session_state["stage"] = "failed"
                        _detach(); st.stop()
                    extracted[lbl] = r
                st.session_state["extracted"] = extracted
                st.write(f"Done — {len(extracted)} documents extracted")

                merged_text   = "\n\n".join(f"=== {k.upper()} ===\n{v['text']}" for k,v in extracted.items())
                merged_tables = sum([v.get("tables",[]) for v in extracted.values()], [])
                merged_discharge = {"text": merged_text, "tables": merged_tables, "status": "success"}

                st.write("**Step 2 / 2** — Running pipeline (translation → completeness → EHR → reporting)")
                init = {
                    "discharge_path":tmp["discharge"],"lab_path":tmp["lab"],"bill_path":tmp["bill"],
                    "extracted_discharge":merged_discharge,"extracted_lab":extracted["lab"],
                    "extracted_bill":extracted["bill"],"normalized":{},"translation":{},
                    "completeness":{},"ehr_validation":{},"report":{},"rag_indexed":False,
                    "current_stage":"starting","status":"running","error":None,
                    "risk_level":None,"hitl_corrections":None,
                }
                result = discharge_pipeline.invoke(init, config=_thread_config())

                for p in tmp.values():
                    try: os.unlink(p)
                    except: pass

                translation = result.get("translation", {})
                st.session_state["translation"] = translation
                if translation.get("status") == "success":
                    nm = translation.get("normalized_output", {})
                    st.write(f"Patient: **{nm.get('patient_name','(not found)')}** "
                             f"· language: `{translation.get('detected_language','?')}` "
                             f"· confidence: `{translation.get('translation_confidence',0):.0%}`")
                elif translation:
                    st.error(f"Translation failed: {translation.get('error', 'unknown error')}")

                graph_snapshot  = discharge_pipeline.get_state(_thread_config())
                is_interrupted  = bool(graph_snapshot.next)
                hitl_payload    = {}

                if is_interrupted:
                    all_interrupts = [
                        intr
                        for task in graph_snapshot.tasks
                        for intr in task.interrupts
                    ]
                    hitl_payload = all_interrupts[0].value if all_interrupts else {}
                    log.info(f"[App] Graph interrupted — payload keys: {list(hitl_payload.keys())}")

                    result = {
                        **result,
                        "completeness": {
                            "passed"         : False,
                            "missing_fields" : hitl_payload.get("missing_fields", []),
                            "rule_violations": hitl_payload.get("rule_violations", []),
                        }
                    }

                st.session_state["pipeline_state"] = result
                st.session_state["hitl_payload"]   = hitl_payload

                if is_interrupted:
                    st.session_state["stage"] = "hitl"
                    st.session_state["_goto_tab"] = 3   # → Results tab (review inline)
                    status_box.update(label="Review required — see the Review tab", state="error")
                    st.warning("Pipeline paused — some clinical data is incomplete. "
                               "Redirecting to the Review tab…")
                elif result.get("status") == "success":
                    st.session_state["stage"] = "done"
                    st.session_state["_goto_tab"] = 3   # → Results tab
                    status_box.update(label="Audit complete", state="complete")
                    st.success(f"Complete — risk: **{result.get('risk_level','?').upper()}**")
                else:
                    st.session_state["stage"] = "failed"
                    status_box.update(label="Failed", state="error")
                    st.error(result.get("error"))

        except Exception as exc:
            import traceback
            st.session_state["stage"] = "failed"
            st.error(f"Unexpected error: {exc}")
            st.code(traceback.format_exc())
            log.exception("[App] crash")
        finally:
            _detach()
        st.rerun()


# ─────────────────────────────────────────────────────────────
#  Shared state
# ─────────────────────────────────────────────────────────────
extracted   = st.session_state.get("extracted")
translation = st.session_state.get("translation")
ps          = st.session_state.get("pipeline_state")


# ═════════════════════════════════════════════════════════════
#  TAB — Extraction
# ═════════════════════════════════════════════════════════════
with tab_extract:
    if not extracted:
        st.markdown(
            '<div class="notice notice-info"><strong>No extraction data yet.</strong><br>'
            '<span>Upload documents and run the pipeline from the Intake tab.</span></div>',
            unsafe_allow_html=True,
        )
    else:
        section_header("Step 01", "Document extraction", "Raw text and tables pulled from each uploaded file.")
        cols = st.columns(len(extracted))
        for i, (lbl, r) in enumerate(extracted.items()):
            with cols[i]:
                with st.container(border=True):
                    st.markdown(f"**{lbl.title()}**")
                    st.markdown('<span class="badge badge-ok">Success</span>', unsafe_allow_html=True)
                    st.caption(f"`{r.get('file_name','—')}`")
                    c_a, c_b = st.columns(2)
                    c_a.metric("Characters", f"{len(r.get('text',''))}")
                    c_b.metric("Tables", len(r.get("tables", [])))
                    if r.get("text"):
                        st.text_area(
                            "Preview",
                            value=r["text"][:500],
                            height=120,
                            disabled=True,
                            key=f"prev_{lbl}",
                        )


# ═════════════════════════════════════════════════════════════
#  TAB — Translation & Normalisation
# ═════════════════════════════════════════════════════════════
with tab_translate:
    if not translation:
        st.markdown(
            '<div class="notice notice-info"><strong>No translation data yet.</strong><br>'
            '<span>Upload documents and run the pipeline from the Intake tab.</span></div>',
            unsafe_allow_html=True,
        )
    else:
        nm   = translation.get("normalized_output", {})
        lang = translation.get("detected_language", "?")
        conf = translation.get("translation_confidence", 0)

        section_header("Step 02", "Translation & Normalisation",
                       "Clinical document translated to English and structured into standard fields.")

        m1, m2, m3 = st.columns(3)
        m1.metric("Language detected", lang.upper())
        m2.metric("Confidence", f"{conf:.0%}")
        m3.metric("Medications", len(nm.get("medications", [])))
        st.markdown('<hr class="rule">', unsafe_allow_html=True)

        left, right = st.columns([3, 2])
        with left:
            st.markdown("**Patient fields**")
            rows = ""
            for f in ALL_PATIENT_FIELDS:
                val = nm.get(f)
                ok  = bool(val) or val == 0
                mark = '<span class="dot dot-ok"></span>' if ok else '<span class="dot dot-danger"></span>'
                disp = (str(val)[:60] + "…" if len(str(val)) > 60 else str(val)) if ok else "Not found"
                vcls = "field-value" if ok else "field-value field-missing"
                rows += (
                    f'<div class="field-row">{mark}'
                    f'<span class="field-label">{FIELD_LABELS[f]}</span>'
                    f'<span class="{vcls}">{disp}</span></div>'
                )
            st.markdown(
                f'<div class="card" style="max-height:420px;overflow-y:auto">{rows}</div>',
                unsafe_allow_html=True,
            )
        with right:
            st.markdown("**Medications prescribed**")
            meds = nm.get("medications", [])
            if not meds:
                st.markdown(
                    '<div class="notice notice-info"><strong>None extracted.</strong></div>',
                    unsafe_allow_html=True,
                )
            for i, med in enumerate(meds, 1):
                with st.container(border=True):
                    st.markdown(f"**{i}. {med.get('medicine_name', '?')}**"
                                f"{'  ' + med.get('strength','') if med.get('strength') else ''}")
                    cols_m = st.columns(3)
                    cols_m[0].caption(f"Dose\n{med.get('dosage','—')}")
                    cols_m[1].caption(f"Freq\n{med.get('frequency','—')}")
                    cols_m[2].caption(f"Route\n{med.get('route','—')}")
                    if med.get("period"):
                        st.caption(f"Duration: {med.get('period')}  ·  Qty: {med.get('total_quantity','—')}")
                    if med.get("remarks"):
                        st.caption(f"Remarks: {med.get('remarks')}")

            st.markdown("<br>", unsafe_allow_html=True)
            lab_vendor = nm.get("lab_vendor")
            labs = nm.get("lab_results", [])
            st.markdown(f"**Lab results**{f' · {lab_vendor}' if lab_vendor else ''}")
            if not labs:
                st.markdown(
                    '<div class="notice notice-info"><strong>None extracted.</strong></div>',
                    unsafe_allow_html=True,
                )
            for lab in labs:
                status = (lab.get("status") or "").lower()
                badge_cls = "badge-danger" if status in ("abnormal", "critical") else "badge-ok"
                with st.container(border=True):
                    c_n, c_s = st.columns([3, 1])
                    c_n.markdown(f"**{lab.get('name', '?')}**")
                    c_s.markdown(f'<span class="badge {badge_cls}">{status or "—"}</span>',
                                 unsafe_allow_html=True)
                    extra = lab.get("unit", "")
                    st.caption(
                        f"Value: {lab.get('value','—')} {extra}"
                        + (f"  ·  Ref: {lab.get('reference_range')}" if lab.get("reference_range") else "")
                    )


# ═════════════════════════════════════════════════════════════
#  TAB — Results  (Completeness → Review → EHR → Audit Report)
# ═════════════════════════════════════════════════════════════
with tab_results:
    if not ps:
        st.markdown(
            '<div class="notice notice-info"><strong>No results yet.</strong><br>'
            '<span>Run the pipeline from the Intake tab to see completeness, '
            'EHR validation, and the audit report here.</span></div>',
            unsafe_allow_html=True,
        )

    if ps:
        comp   = ps.get("completeness", {})
        ehr    = ps.get("ehr_validation", {})
        report = ps.get("report", {})

        hitl_payload_for_display = st.session_state.get("hitl_payload", {})
        comp_to_show = comp if comp else (
            {
                "passed"         : False,
                "missing_fields" : hitl_payload_for_display.get("missing_fields", []),
                "rule_violations": hitl_payload_for_display.get("rule_violations", []),
            }
            if hitl_payload_for_display else None
        )

        # ── Step 03: Completeness ─────────────────────────────
        if comp_to_show is not None:
            passed  = comp_to_show.get("passed", False)
            missing = comp_to_show.get("missing_fields", [])
            viols   = comp_to_show.get("rule_violations", [])

            section_header("Step 03", "Completeness check")
            badge = ('<span class="badge badge-ok">Passed</span>' if passed
                     else f'<span class="badge badge-warn">{len(missing)} missing · '
                          f'{len(viols)} violation(s)</span>')
            st.markdown(badge, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            if missing or viols:
                with st.container(border=True):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown('<span style="color:var(--ink-soft);font-size:.78rem;'
                                    'text-transform:uppercase;letter-spacing:.05em">Missing / Violations</span>',
                                    unsafe_allow_html=True)
                        for f in missing:
                            st.markdown(f'<div class="field-row"><span class="dot dot-danger"></span>'
                                        f'<span style="color:var(--danger)">{FIELD_LABELS.get(f,f)}</span></div>',
                                        unsafe_allow_html=True)
                        for v in viols:
                            st.markdown(f'<div class="field-row"><span class="dot dot-warn"></span>'
                                        f'<span style="color:var(--warn);font-size:.82rem">{v}</span></div>',
                                        unsafe_allow_html=True)
                    with c2:
                        st.markdown('<span style="color:var(--ink-soft);font-size:.78rem;'
                                    'text-transform:uppercase;letter-spacing:.05em">Present</span>',
                                    unsafe_allow_html=True)
                        for f in [f for f in ALL_PATIENT_FIELDS if f not in missing]:
                            st.markdown(f'<div class="field-row"><span class="dot dot-ok"></span>'
                                        f'<span>{FIELD_LABELS.get(f,f)}</span></div>',
                                        unsafe_allow_html=True)
            st.markdown('<hr class="rule">', unsafe_allow_html=True)

        # ── Step 04: Human review (inline HITL) ──────────────
        if st.session_state["stage"] == "hitl":
            payload  = st.session_state.get("hitl_payload", {})
            missing  = payload.get("missing_fields", [])
            viols    = payload.get("rule_violations", [])
            nm       = payload.get("normalized",
                                   st.session_state.get("pipeline_state", {}).get("normalized", {}))

            section_header("Step 04", "Human review", "Complete missing clinical data before validation continues.")
            st.markdown(
                '<div class="notice notice-warn"><strong>Some clinical data could not be extracted.</strong><br>'
                '<span>Fill in what you know — fields left blank are fine. '
                'Choose Continue or Skip to proceed.</span></div>',
                unsafe_allow_html=True
            )

            with st.form("hitl_form"):
                corrections = {}

                demo = [f for f in ["patient_id","patient_name","age","gender","address"] if f in missing]
                if demo:
                    st.markdown("**Demographics**")
                    c1, c2 = st.columns(2)
                    with c1:
                        if "patient_id" in missing:
                            v = st.text_input("Patient ID", placeholder="e.g. P1008")
                            if v: corrections["patient_id"] = v
                        if "patient_name" in missing:
                            v = st.text_input("Patient Name")
                            if v: corrections["patient_name"] = v
                        if "age" in missing:
                            v = st.text_input("Age", placeholder="e.g. 45")
                            if v: corrections["age"] = v
                    with c2:
                        if "gender" in missing:
                            v = st.selectbox("Gender", ["","Male","Female","Other","Unknown"])
                            if v: corrections["gender"] = v
                        if "address" in missing:
                            v = st.text_area("Address", height=80)
                            if v: corrections["address"] = v

                admit = [f for f in ["admission_date","discharge_date","ward","bed_no"] if f in missing]
                if admit:
                    st.markdown("**Admission**")
                    c1, c2 = st.columns(2)
                    with c1:
                        if "admission_date" in missing:
                            v = st.text_input("Admission Date (YYYY-MM-DD)", placeholder="2025-06-01")
                            if v: corrections["admission_date"] = v
                        if "ward" in missing:
                            v = st.text_input("Ward", placeholder="e.g. General / ICU")
                            if v: corrections["ward"] = v
                    with c2:
                        if "discharge_date" in missing:
                            v = st.text_input("Discharge Date (YYYY-MM-DD)", placeholder="2025-06-07")
                            if v: corrections["discharge_date"] = v
                        if "bed_no" in missing:
                            v = st.text_input("Bed No.", placeholder="e.g. B-12")
                            if v: corrections["bed_no"] = v

                team = [f for f in ["attending_physician","consulting_doctors"] if f in missing]
                if team:
                    st.markdown("**Clinical team**")
                    c1, c2 = st.columns(2)
                    with c1:
                        if "attending_physician" in missing:
                            v = st.text_input("Attending Physician")
                            if v: corrections["attending_physician"] = v
                    with c2:
                        if "consulting_doctors" in missing:
                            raw = st.text_area("Consulting Doctors (one per line)", height=80)
                            parsed = [x.strip() for x in raw.splitlines() if x.strip()]
                            if parsed: corrections["consulting_doctors"] = parsed

                if "allergies" in missing:
                    st.markdown("**Allergies**")
                    raw = st.text_input("Allergies (comma-separated, or 'None')",
                                        placeholder="Penicillin, Aspirin — or None")
                    if raw.strip():
                        corrections["allergies"] = ([] if raw.strip().lower()=="none"
                                                    else [x.strip() for x in raw.split(",") if x.strip()])

                notes_f = [f for f in ["discharge_diagnosis","follow_up_appointment","discharge_instructions"] if f in missing]
                if notes_f:
                    st.markdown("**Clinical notes**")
                    if "discharge_diagnosis" in missing:
                        v = st.text_area("Discharge Diagnosis", height=70)
                        if v: corrections["discharge_diagnosis"] = v
                    c1, c2 = st.columns(2)
                    with c1:
                        if "follow_up_appointment" in missing:
                            v = st.text_input("Follow-up Appointment",
                                              placeholder="e.g. 2 weeks — Cardiology OPD")
                            if v: corrections["follow_up_appointment"] = v
                    with c2:
                        if "discharge_instructions" in missing:
                            v = st.text_area("Discharge Instructions", height=70)
                            if v: corrections["discharge_instructions"] = v

                if viols:
                    st.markdown("**Prescription details**")
                    st.caption("Edit any incomplete rows. Blank cells are fine.")
                    meds = [dict(m) for m in nm.get("medications", [])]
                    for i, med in enumerate(meds, 1):
                        for k in PRESCRIPTION_FIELDS:
                            med.setdefault(k, "")
                        if not med.get("sl_no"): med["sl_no"] = str(i)
                    if not meds:
                        meds = [{k:"" for k in PRESCRIPTION_FIELDS}]
                        meds[0]["sl_no"] = "1"

                    edited = st.data_editor(
                        meds, num_rows="dynamic", use_container_width=True,
                        column_config={
                            "sl_no"         : st.column_config.TextColumn("Sl.", width="small"),
                            "medicine_name" : st.column_config.TextColumn("Medicine", width="medium"),
                            "strength"      : st.column_config.TextColumn("Strength", width="small"),
                            "dosage"        : st.column_config.TextColumn("Dosage", width="small"),
                            "frequency"     : st.column_config.TextColumn("Frequency", width="medium"),
                            "route"         : st.column_config.TextColumn("Route", width="small"),
                            "period"        : st.column_config.TextColumn("Period", width="small"),
                            "remarks"       : st.column_config.TextColumn("Remarks", width="medium"),
                            "total_quantity": st.column_config.TextColumn("Qty", width="small"),
                        }
                    )
                    corrections["medications"] = [dict(r) for r in edited]

                st.markdown("<br>", unsafe_allow_html=True)
                col1, col2, _ = st.columns([1, 1, 5])
                with col1:
                    submit  = st.form_submit_button("Continue pipeline", type="primary")
                with col2:
                    skip_it = st.form_submit_button("Skip — use as-is")

            if submit or skip_it:
                data_to_resume = corrections if submit else {}
                log.info(f"[App] HITL {'corrections' if submit else 'skip'} — "
                         f"keys: {list(data_to_resume.keys())}")
                _attach()
                try:
                    with st.spinner("Resuming pipeline…"):
                        result = discharge_pipeline.invoke(
                            Command(resume=data_to_resume),
                            config=_thread_config(),
                        )

                    graph_snapshot = discharge_pipeline.get_state(_thread_config())
                    is_interrupted = bool(graph_snapshot.next)

                    if is_interrupted:
                        all_interrupts = [
                            intr for task in graph_snapshot.tasks for intr in task.interrupts
                        ]
                        new_payload = all_interrupts[0].value if all_interrupts else {}
                        st.session_state["hitl_payload"] = new_payload
                        result = {
                            **result,
                            "completeness": {
                                "passed"         : False,
                                "missing_fields" : new_payload.get("missing_fields", []),
                                "rule_violations": new_payload.get("rule_violations", []),
                            }
                        }
                        st.session_state["stage"] = "hitl"
                    elif result.get("status") == "success":
                        st.session_state["stage"] = "done"
                        st.session_state["hitl_payload"] = {}
                        st.session_state["_goto_tab"] = 3   # → Results tab
                    else:
                        st.session_state["stage"] = "failed"

                    st.session_state["pipeline_state"] = result

                except Exception as exc:
                    import traceback
                    st.session_state["stage"] = "failed"
                    st.error(f"Resume failed: {exc}")
                    st.code(traceback.format_exc())
                    log.exception("[App] resume crash")
                finally:
                    _detach()
                st.rerun()

            st.markdown('<hr class="rule">', unsafe_allow_html=True)

        # ── Step 05: EHR validation ───────────────────────────
        if ehr:
            disc   = ehr.get("discrepancies", [])
            passed = ehr.get("passed", False)
            section_header("Step 05", "EHR validation")
            badge = ('<span class="badge badge-ok">Passed</span>' if passed
                     else f'<span class="badge badge-danger">{len(disc)} discrepancy(ies)</span>')
            st.markdown(badge, unsafe_allow_html=True)
            if disc:
                st.markdown("<br>", unsafe_allow_html=True)
                for d in disc:
                    st.markdown(
                        f'<div class="card card-tight" style="border-left:3px solid var(--danger)">'
                        f'<span class="badge badge-danger">{d.get("rule","—")}</span>'
                        f'&nbsp; {d.get("detail", str(d))}</div>',
                        unsafe_allow_html=True
                    )
            for c in ehr.get("allergy_conflicts", []):
                st.error(f"Allergy conflict: {c}")
            st.markdown('<hr class="rule">', unsafe_allow_html=True)

        if report and ps.get("status") == "success":
            nm_final  = ps.get("normalized", {})
            hitl_corr = ps.get("hitl_corrections") or {}
            risk  = report.get("risk_level", "unknown")
            score = report.get("risk_score", 0)
            rec   = report.get("recommendation", "—")
            rc    = {"low":"risk-low","medium":"risk-medium","high":"risk-high"}.get(risk,"risk-low")
            pid   = nm_final.get("patient_id", "unknown")

            section_header("Step 06", "Audit report")
            st.markdown(
                f'<div class="risk-panel {rc}">'
                f'<div class="risk-label">Risk: {risk.upper()}</div>'
                f'<div class="risk-meta">Score: <strong>{score}</strong>'
                f'&nbsp;&nbsp;·&nbsp;&nbsp;{rec}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

            report_files = report.get("report_files") or {}
            if report_files:
                saved = "  ·  ".join(f"{fmt}: `{path}`" for fmt, path in report_files.items())
                st.caption(f"Saved to disk — {saved}")

            t1, t2, t3, t4, t5 = st.tabs(
                ["Summary", "Risk Flags", "Full JSON", "Download JSON", "Download PDF"]
            )
            with t1:
                def _v(val):
                    return str(val) if val not in (None, "", [], {}) else "—"

                # ── Patient header card ──────────────────────────────
                risk_colors = {
                    "low":    ("var(--ok-bg)",      "var(--ok)",     "var(--ok-line)"),
                    "medium": ("var(--warn-bg)",     "var(--warn)",   "var(--warn-line)"),
                    "high":   ("var(--danger-bg)",   "var(--danger)", "var(--danger-line)"),
                }
                rb, rt, rl = risk_colors.get(risk, ("var(--neutral-bg)", "var(--neutral)", "var(--neutral-line)"))
                st.markdown(
                    f'<div class="card" style="margin-bottom:14px">'
                    f'<div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:12px">'
                    f'<div>'
                    f'<div style="font-size:1.25rem;font-weight:700;color:var(--ink)">'
                    f'{_v(nm_final.get("patient_name"))}</div>'
                    f'<div style="font-size:.82rem;color:var(--ink-soft);margin-top:2px">'
                    f'ID: {_v(nm_final.get("patient_id"))}'
                    f'&ensp;·&ensp;{_v(nm_final.get("age"))} yrs'
                    f'&ensp;·&ensp;{_v(nm_final.get("gender"))}'
                    f'</div>'
                    f'</div>'
                    f'<div style="background:{rb};color:{rt};border:1px solid {rl};'
                    f'border-radius:6px;padding:6px 14px;font-weight:700;font-size:.9rem;'
                    f'text-align:center;min-width:100px">'
                    f'RISK: {risk.upper()}<br>'
                    f'<span style="font-size:.75rem;font-weight:400">Score {score}</span>'
                    f'</div>'
                    f'</div>'
                    f'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px 24px;'
                    f'margin-top:14px;padding-top:12px;border-top:1px solid var(--border)">'
                    f'<div><span style="font-size:.72rem;color:var(--ink-soft);text-transform:uppercase;'
                    f'letter-spacing:.05em">Ward / Bed</span>'
                    f'<div style="font-size:.88rem;font-weight:600">{_v(nm_final.get("ward"))} / {_v(nm_final.get("bed_no"))}</div></div>'
                    f'<div><span style="font-size:.72rem;color:var(--ink-soft);text-transform:uppercase;'
                    f'letter-spacing:.05em">Admitted</span>'
                    f'<div style="font-size:.88rem;font-weight:600">{_v(nm_final.get("admission_date"))}</div></div>'
                    f'<div><span style="font-size:.72rem;color:var(--ink-soft);text-transform:uppercase;'
                    f'letter-spacing:.05em">Discharged</span>'
                    f'<div style="font-size:.88rem;font-weight:600">{_v(nm_final.get("discharge_date"))}</div></div>'
                    f'<div><span style="font-size:.72rem;color:var(--ink-soft);text-transform:uppercase;'
                    f'letter-spacing:.05em">Attending</span>'
                    f'<div style="font-size:.88rem;font-weight:600">{_v(nm_final.get("attending_physician"))}</div></div>'
                    f'<div><span style="font-size:.72rem;color:var(--ink-soft);text-transform:uppercase;'
                    f'letter-spacing:.05em">Service Line</span>'
                    f'<div style="font-size:.88rem;font-weight:600">{_v(nm_final.get("service_line"))}</div></div>'
                    f'<div><span style="font-size:.72rem;color:var(--ink-soft);text-transform:uppercase;'
                    f'letter-spacing:.05em">Allergies</span>'
                    f'<div style="font-size:.88rem;font-weight:600">'
                    f'{", ".join(nm_final.get("allergies") or []) or "None"}</div></div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

                # ── Diagnosis ────────────────────────────────────────
                st.markdown(
                    f'<div style="font-size:.75rem;font-weight:700;text-transform:uppercase;'
                    f'letter-spacing:.08em;color:var(--ink-soft);margin:16px 0 6px">Discharge Diagnosis</div>'
                    f'<div class="card" style="font-size:.92rem;line-height:1.65;margin-bottom:14px">'
                    f'{_v(nm_final.get("discharge_diagnosis"))}</div>',
                    unsafe_allow_html=True,
                )

                # ── Medications ──────────────────────────────────────
                meds_sum = nm_final.get("medications") or []
                st.markdown(
                    f'<div style="font-size:.75rem;font-weight:700;text-transform:uppercase;'
                    f'letter-spacing:.08em;color:var(--ink-soft);margin-bottom:8px">'
                    f'Medications Prescribed ({len(meds_sum)})</div>',
                    unsafe_allow_html=True,
                )
                if not meds_sum:
                    st.markdown(
                        '<div class="notice notice-info" style="margin-bottom:14px">'
                        '<strong>No medications recorded.</strong></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    rows_html = ""
                    for i, med in enumerate(meds_sum, 1):
                        bg = "background:var(--surface-2)" if i % 2 == 0 else ""
                        rows_html += (
                            f'<tr style="{bg}">'
                            f'<td style="padding:7px 10px;font-weight:600;color:var(--accent)">{i}</td>'
                            f'<td style="padding:7px 10px;font-weight:600">{_v(med.get("medicine_name"))}'
                            f'{"  <small style=\'color:var(--ink-soft)\'>" + med.get("strength","") + "</small>" if med.get("strength") else ""}</td>'
                            f'<td style="padding:7px 10px">{_v(med.get("dosage"))}</td>'
                            f'<td style="padding:7px 10px">{_v(med.get("frequency"))}</td>'
                            f'<td style="padding:7px 10px">{_v(med.get("route"))}</td>'
                            f'<td style="padding:7px 10px">{_v(med.get("period"))}</td>'
                            f'<td style="padding:7px 10px;color:var(--ink-soft);font-size:.82rem">{_v(med.get("remarks"))}</td>'
                            f'</tr>'
                        )
                    st.markdown(
                        f'<div class="card" style="padding:0;overflow:hidden;margin-bottom:14px">'
                        f'<table style="width:100%;border-collapse:collapse;font-size:.85rem">'
                        f'<thead><tr style="background:var(--surface-2);border-bottom:2px solid var(--border)">'
                        f'<th style="padding:8px 10px;text-align:left;font-size:.72rem;text-transform:uppercase;'
                        f'letter-spacing:.06em;color:var(--ink-soft);white-space:nowrap">#</th>'
                        f'<th style="padding:8px 10px;text-align:left;font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;color:var(--ink-soft)">Medicine</th>'
                        f'<th style="padding:8px 10px;text-align:left;font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;color:var(--ink-soft)">Dosage</th>'
                        f'<th style="padding:8px 10px;text-align:left;font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;color:var(--ink-soft)">Frequency</th>'
                        f'<th style="padding:8px 10px;text-align:left;font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;color:var(--ink-soft)">Route</th>'
                        f'<th style="padding:8px 10px;text-align:left;font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;color:var(--ink-soft)">Period</th>'
                        f'<th style="padding:8px 10px;text-align:left;font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;color:var(--ink-soft)">Remarks</th>'
                        f'</tr></thead>'
                        f'<tbody>{rows_html}</tbody>'
                        f'</table></div>',
                        unsafe_allow_html=True,
                    )

                # ── Lab reports ────────────────────────────────────
                labs_sum  = nm_final.get("lab_results") or []
                lab_vendor_sum = nm_final.get("lab_vendor")
                st.markdown(
                    f'<div style="font-size:.75rem;font-weight:700;text-transform:uppercase;'
                    f'letter-spacing:.08em;color:var(--ink-soft);margin-bottom:8px">'
                    f'Lab Reports ({len(labs_sum)})'
                    f'{f" &middot; {_v(lab_vendor_sum)}" if lab_vendor_sum else ""}</div>',
                    unsafe_allow_html=True,
                )
                if not labs_sum:
                    st.markdown(
                        '<div class="notice notice-info" style="margin-bottom:14px">'
                        '<strong>No lab results recorded.</strong></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    lab_rows_html = ""
                    for i, lab in enumerate(labs_sum, 1):
                        status = (lab.get("status") or "").lower()
                        scls = "var(--danger)" if status in ("abnormal", "critical") else "var(--ok)"
                        bg = "background:var(--surface-2)" if i % 2 == 0 else ""
                        lab_rows_html += (
                            f'<tr style="{bg}">'
                            f'<td style="padding:7px 10px;font-weight:600">{_v(lab.get("name"))}</td>'
                            f'<td style="padding:7px 10px">{_v(lab.get("value"))} {_v(lab.get("unit")) if lab.get("unit") else ""}</td>'
                            f'<td style="padding:7px 10px;color:var(--ink-soft);font-size:.82rem">{_v(lab.get("reference_range"))}</td>'
                            f'<td style="padding:7px 10px;font-weight:600;color:{scls}">{_v(lab.get("status")).upper()}</td>'
                            f'</tr>'
                        )
                    st.markdown(
                        f'<div class="card" style="padding:0;overflow:hidden;margin-bottom:14px">'
                        f'<table style="width:100%;border-collapse:collapse;font-size:.85rem">'
                        f'<thead><tr style="background:var(--surface-2);border-bottom:2px solid var(--border)">'
                        f'<th style="padding:8px 10px;text-align:left;font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;color:var(--ink-soft)">Test</th>'
                        f'<th style="padding:8px 10px;text-align:left;font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;color:var(--ink-soft)">Value</th>'
                        f'<th style="padding:8px 10px;text-align:left;font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;color:var(--ink-soft)">Reference Range</th>'
                        f'<th style="padding:8px 10px;text-align:left;font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;color:var(--ink-soft)">Status</th>'
                        f'</tr></thead>'
                        f'<tbody>{lab_rows_html}</tbody>'
                        f'</table></div>',
                        unsafe_allow_html=True,
                    )

                # ── Care plan: instructions + follow-up ──────────────
                ci1, ci2 = st.columns(2)
                with ci1:
                    st.markdown(
                        f'<div style="font-size:.75rem;font-weight:700;text-transform:uppercase;'
                        f'letter-spacing:.08em;color:var(--ink-soft);margin-bottom:6px">Discharge Instructions</div>'
                        f'<div class="card" style="font-size:.88rem;line-height:1.6;min-height:72px">'
                        f'{_v(nm_final.get("discharge_instructions"))}</div>',
                        unsafe_allow_html=True,
                    )
                with ci2:
                    followup = _v(nm_final.get("follow_up_appointment"))
                    fu_color = "var(--ok)" if nm_final.get("follow_up_appointment") else "var(--warn)"
                    st.markdown(
                        f'<div style="font-size:.75rem;font-weight:700;text-transform:uppercase;'
                        f'letter-spacing:.08em;color:var(--ink-soft);margin-bottom:6px">Follow-up Appointment</div>'
                        f'<div class="card" style="font-size:.88rem;font-weight:600;color:{fu_color};'
                        f'min-height:72px;display:flex;align-items:center">{followup}</div>',
                        unsafe_allow_html=True,
                    )

                # ── Billing ──────────────────────────────────────────
                bill_paid    = nm_final.get("bill_paid", False)
                discharge_ok = nm_final.get("discharge_ok", False)
                st.markdown(
                    f'<div style="font-size:.75rem;font-weight:700;text-transform:uppercase;'
                    f'letter-spacing:.08em;color:var(--ink-soft);margin:16px 0 6px">Billing & Release</div>'
                    f'<div class="card card-tight" style="display:flex;gap:32px;align-items:center">'
                    f'<div><span style="font-size:.78rem;color:var(--ink-soft)">Total Bill</span>'
                    f'<div style="font-size:1.1rem;font-weight:700">{_v(nm_final.get("total_bill"))}</div></div>'
                    f'<div><span style="font-size:.78rem;color:var(--ink-soft)">Payment Status</span>'
                    f'<div style="font-size:.95rem;font-weight:700;'
                    f'color:{"var(--ok)" if bill_paid else "var(--danger)"}">'
                    f'{"Paid" if bill_paid else "Outstanding"}</div></div>'
                    f'<div><span style="font-size:.78rem;color:var(--ink-soft)">Discharge Release</span>'
                    f'<div style="font-size:.95rem;font-weight:700;'
                    f'color:{"var(--ok)" if discharge_ok else "var(--danger)"}">'
                    f'{"Released" if discharge_ok else "Not approved"}</div></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # ── HITL corrections applied ──────────────────────────
                if hitl_corr:
                    st.markdown(
                        f'<div style="font-size:.75rem;font-weight:700;text-transform:uppercase;'
                        f'letter-spacing:.08em;color:var(--ink-soft);margin:16px 0 6px">'
                        f'Reviewer Corrections Applied</div>',
                        unsafe_allow_html=True,
                    )
                    rows_h = "".join(
                        f'<tr><td style="padding:5px 10px;font-weight:600;font-size:.83rem;'
                        f'color:var(--accent);white-space:nowrap">{k.replace("_"," ").title()}</td>'
                        f'<td style="padding:5px 10px;font-size:.83rem">'
                        f'{", ".join(str(x) for x in v) if isinstance(v, list) else str(v)}</td></tr>'
                        for k, v in hitl_corr.items()
                    )
                    st.markdown(
                        f'<div class="card" style="padding:0;overflow:hidden;margin-bottom:8px;'
                        f'border-left:3px solid var(--accent)">'
                        f'<table style="width:100%;border-collapse:collapse">'
                        f'<tbody>{rows_h}</tbody></table></div>',
                        unsafe_allow_html=True,
                    )

                # ── Recommendation footer ────────────────────────────
                st.markdown(
                    f'<div class="card card-tight" style="background:{rb};border-color:{rl};'
                    f'margin-top:10px;display:flex;align-items:center;gap:12px">'
                    f'<span style="font-size:1.4rem">{"✅" if risk=="low" else ("⚠️" if risk=="medium" else "🚨")}</span>'
                    f'<div><div style="font-size:.78rem;color:{rt};font-weight:700;'
                    f'text-transform:uppercase;letter-spacing:.05em">Recommendation</div>'
                    f'<div style="font-size:.9rem;font-weight:600;color:{rt}">{_v(rec)}</div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

            with t2:
                flags = report.get("flags", [])
                if not flags:
                    st.markdown(
                        '<div class="notice notice-info"><strong>No risk flags.</strong></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption(f"{len(flags)} flag(s) — sorted by severity")
                    for flag in sorted(flags, key=lambda x: -x.get("weight", 0)):
                        w    = flag.get("weight", 0)
                        sev  = "danger" if w >= 8 else ("warn" if w >= 3 else "info")
                        st.markdown(
                            f'<div class="card card-tight" style="border-left:3px solid var(--{sev})">'
                            f'<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">'
                            f'<span class="badge badge-{sev}">{flag.get("rule","—")}</span>'
                            f'<span style="font-family:IBM Plex Mono,monospace;font-size:.8rem;'
                            f'color:var(--{sev});font-weight:700">+{w}</span>'
                            f'<span style="color:var(--ink-soft);font-size:.86rem;flex:1">'
                            f'{flag.get("detail","—")}</span>'
                            f'</div></div>',
                            unsafe_allow_html=True
                        )
            with t3:
                st.json(report)
            with t4:
                st.markdown(
                    '<div class="notice notice-info" style="margin-bottom:16px">'
                    '<strong>JSON report</strong><br>'
                    '<span>Machine-readable full audit output including flags, '
                    'audit trail, and rules version.</span></div>',
                    unsafe_allow_html=True,
                )
                st.download_button(
                    "Download audit_report.json",
                    data=json.dumps(report, indent=2, default=str),
                    file_name=f"audit_{pid}.json",
                    mime="application/json",
                    use_container_width=True,
                )
            with t5:
                st.markdown(
                    '<div class="notice notice-info" style="margin-bottom:16px">'
                    '<strong>PDF report</strong><br>'
                    '<span>Formatted discharge audit summary with risk level, '
                    'flags, and recommendation — ready to print or share.</span></div>',
                    unsafe_allow_html=True,
                )
                nm_pdf = {**nm_final, **({"_hitl_corrections": hitl_corr} if hitl_corr else {})}
                pdf_bytes = generate_pdf_report(report, nm_pdf)
                st.download_button(
                    "Download audit_report.pdf",
                    data=pdf_bytes,
                    file_name=f"audit_{pid}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )




# ═════════════════════════════════════════════════════════════
#  TAB — Ask records (RAG Q&A)
# ═════════════════════════════════════════════════════════════
_INDEX_PATH = ROOT / "Data" / "faiss_index" / "index.faiss"

with tab_ask:
    if not _INDEX_PATH.exists():
        st.markdown(
            '<div class="notice notice-info"><strong>No indexed records yet.</strong><br>'
            '<span>Run the pipeline at least once to build the patient record index.</span></div>',
            unsafe_allow_html=True,
        )
    else:
        from agents.rag_agents.indexing_agent import list_indexed_patients
        from agents.rag_agents.rag_graph import rag_app

        section_header("Records", "Ask about patient records")

        patients = list_indexed_patients()
        current_patient_id = ps.get("normalized", {}).get("patient_id") if ps else None

        options_map = {"All patients": None}
        for pid, info in patients.items():
            label = f"{info.get('patient_name', pid)} ({pid})"
            options_map[label] = pid

        option_labels = list(options_map.keys())

        default_idx = 0
        if current_patient_id:
            for i, lbl in enumerate(option_labels):
                if options_map[lbl] == current_patient_id:
                    default_idx = i
                    break

        selected_label = st.selectbox(
            "Scope queries to",
            option_labels,
            index=default_idx,
            key="rag_patient_selector",
        )
        selected_patient_id = options_map[selected_label]

        if st.session_state.get("qa_patient_scope") != selected_patient_id:
            st.session_state["qa_history"]       = []
            st.session_state["qa_patient_scope"] = selected_patient_id

        if selected_patient_id:
            st.caption(f"Querying records scoped to {selected_label}.")
        else:
            st.caption("Querying across all indexed patient records.")

        for q, a in st.session_state.get("qa_history", []):
            st.chat_message("user").write(q)
            st.chat_message("assistant").write(a)

        question = st.chat_input("Ask a question about patient records…")
        if question:
            st.chat_message("user").write(question)

            with st.spinner("Searching records…"):
                init_state = {
                    "question"   : question,
                    "patient_id" : selected_patient_id,
                    "documents"  : [],
                    "generation" : "",
                    "is_grounded": False,
                    "loop_count" : 0,
                    "scores"     : {},
                }
                try:
                    final_state = rag_app.invoke(init_state)
                    answer = final_state.get("generation", "I'm sorry, I couldn't find an answer.")
                except Exception as e:
                    log.exception("RAG Q&A error")
                    answer = f"Error generating answer: {e}"

            st.chat_message("assistant").write(answer)
            st.session_state["qa_history"].append((question, answer))
            st.rerun()


# ═════════════════════════════════════════════════════════════
#  TAB — Logs
# ═════════════════════════════════════════════════════════════
with tab_logs:
    section_header("Diagnostics", "Pipeline logs")
    if st.session_state.get("pipeline_logs"):
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.code("\n".join(st.session_state["pipeline_logs"]), language="text")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="notice notice-info"><strong>No logs yet.</strong><br>'
            '<span>Logs appear here once you run or resume the pipeline.</span></div>',
            unsafe_allow_html=True,
        )