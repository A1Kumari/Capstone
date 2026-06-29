"""
ui/streamlit_app.py — DischargeAI Clinical Auditor (single-page)
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
from agents.translation_normalization_agent import translate_and_normalize
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


# ─────────────────────────────────────────────────────────────
#  Page config & CSS
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DischargeAI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background:#f8fafc; }
[data-testid="stSidebar"]          { background:#1e293b; }
[data-testid="stSidebar"] *        { color:#e2e8f0 !important; }
[data-testid="stSidebar"] hr       { border-color:#334155; }

.badge { display:inline-block; padding:2px 10px; border-radius:20px;
         font-size:.75rem; font-weight:700; letter-spacing:.04em; text-transform:uppercase; }
.badge-ok      { background:#dcfce7; color:#166534; border:1px solid #86efac; }
.badge-warn    { background:#fef9c3; color:#854d0e; border:1px solid #fde047; }
.badge-danger  { background:#fee2e2; color:#991b1b; border:1px solid #fca5a5; }
.badge-info    { background:#dbeafe; color:#1e40af; border:1px solid #93c5fd; }
.badge-neutral { background:#f1f5f9; color:#475569; border:1px solid #cbd5e1; }

.field-row { display:flex; align-items:center; gap:8px; padding:5px 0;
             border-bottom:1px solid #f1f5f9; font-size:.875rem; }
.tick  { color:#16a34a; font-weight:700; min-width:16px; }
.cross { color:#dc2626; font-weight:700; min-width:16px; }

.result-card { background:white; border-radius:10px; padding:14px 18px;
               border:1px solid #e2e8f0; margin-bottom:10px;
               box-shadow:0 1px 3px rgba(0,0,0,.04); }

.risk-low    { background:#f0fdf4; border-left:4px solid #22c55e; padding:12px 18px; border-radius:0 8px 8px 0; }
.risk-medium { background:#fefce8; border-left:4px solid #eab308; padding:12px 18px; border-radius:0 8px 8px 0; }
.risk-high   { background:#fef2f2; border-left:4px solid #ef4444; padding:12px 18px; border-radius:0 8px 8px 0; }

.hitl-box { background:#fff7ed; border:1px solid #fed7aa; border-left:4px solid #f97316;
            border-radius:0 10px 10px 0; padding:16px 20px; margin:16px 0; }
</style>
""", unsafe_allow_html=True)


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
    st.markdown("## 🏥 DischargeAI")
    st.caption("Clinical Discharge Auditor")
    st.divider()

    STAGE_BADGE = {
        "upload": ("⬜", "Ready",            "neutral"),
        "hitl"  : ("🟡", "Review Required",  "warn"),
        "done"  : ("🟢", "Complete",          "ok"),
        "failed": ("🔴", "Failed",            "danger"),
    }
    icon, label, kind = STAGE_BADGE.get(stage, ("⬜", "—", "neutral"))
    st.markdown(f"**Status:** {icon} "
                f'<span class="badge badge-{kind}">{label}</span>',
                unsafe_allow_html=True)

    if ps and ps.get("risk_level"):
        risk = ps["risk_level"]
        rc   = {"low":"ok","medium":"warn","high":"danger"}.get(risk,"neutral")
        st.markdown(f'**Risk:** <span class="badge badge-{rc}">{risk.upper()}</span>',
                    unsafe_allow_html=True)

    st.divider()
    if st.button("🔄 New Case", use_container_width=True):
        _reset()
        st.rerun()


# ─────────────────────────────────────────────────────────────
#  Main area
# ─────────────────────────────────────────────────────────────
st.markdown("# 🏥 DischargeAI — Clinical Auditor")
st.caption("Upload three discharge documents and run the full audit pipeline.")
st.markdown("---")

from fpdf import FPDF

def generate_pdf_report(report):
    pdf = FPDF()
    pdf.set_margins(15, 15, 15)
    pdf.add_page()
    _W = pdf.w - pdf.l_margin - pdf.r_margin  # usable width, computed once

    def _safe(value) -> str:
        """Sanitise any value to a latin-1-safe string for FPDF core fonts."""
        return str(value).encode("latin-1", "replace").decode("latin-1")

    def _body(text: str):
        """Write a paragraph, always starting from the left margin."""
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(_W, 7, text=_safe(text))

    def _heading(text: str, size: int = 12):
        pdf.set_x(pdf.l_margin)
        pdf.set_font("helvetica", "B", size)
        pdf.cell(_W, 10, text=_safe(text), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("helvetica", size=11)

    # ── Title ─────────────────────────────────────────────────
    pdf.set_font("helvetica", "B", 16)
    pdf.cell(_W, 10, text="Clinical Audit Report", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(8)

    # ── Summary ───────────────────────────────────────────────
    _heading("Summary")
    _body(report.get("summary") or "No summary available.")
    pdf.ln(4)

    # ── Risk ──────────────────────────────────────────────────
    _heading(f"Risk Level: {(report.get('risk_level') or 'Unknown').upper()}")
    _heading(f"Risk Score: {report.get('risk_score', 0)}")
    _body(f"Recommendation: {report.get('recommendation') or 'None'}")
    pdf.ln(4)

    # ── Flags ─────────────────────────────────────────────────
    _heading("Audit Flags")
    flags = report.get("flags") or []
    if not flags:
        _body("No flags raised.")
    else:
        for flag in flags:
            rule   = flag.get("rule", "")
            weight = flag.get("weight", 0)
            detail = flag.get("detail", "")
            _body(f"[{rule}] (+{weight})  {detail}")
            pdf.ln(1)

    return bytes(pdf.output())

# ═════════════════════════════════════════════════════════════
#  SECTION 1 — Upload & Run
# ═════════════════════════════════════════════════════════════
st.markdown("### 📤 Upload Documents")
c1, c2, c3 = st.columns(3)
with c1:
    st.caption("**Discharge Report**  *(doctor's summary)*")
    discharge_file = st.file_uploader("Discharge", label_visibility="collapsed",
        type=["pdf","docx","png","jpg","jpeg","txt","json"], key="up_d")
    if discharge_file:
        st.markdown('<span class="badge badge-ok">✓ Ready</span>', unsafe_allow_html=True)
with c2:
    st.caption("**Lab Report**  *(test results)*")
    lab_file = st.file_uploader("Lab", label_visibility="collapsed",
        type=["pdf","docx","png","jpg","jpeg","txt","json"], key="up_l")
    if lab_file:
        st.markdown('<span class="badge badge-ok">✓ Ready</span>', unsafe_allow_html=True)
with c3:
    st.caption("**Hospital Bill**  *(billing summary)*")
    bill_file = st.file_uploader("Bill", label_visibility="collapsed",
        type=["pdf","docx","png","jpg","jpeg","txt","json"], key="up_b")
    if bill_file:
        st.markdown('<span class="badge badge-ok">✓ Ready</span>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

run_ready = all([discharge_file, lab_file, bill_file])
if not run_ready:
    st.caption(f"📎 {sum(1 for f in [discharge_file,lab_file,bill_file] if f)}/3 files uploaded")

if st.button("▶  Run Audit Pipeline", type="primary", disabled=not run_ready):
    _attach()
    tmp = {}
    for lbl, f in [("discharge",discharge_file),("lab",lab_file),("bill",bill_file)]:
        t = tempfile.NamedTemporaryFile(delete=False, suffix=Path(f.name).suffix)
        t.write(f.read()); t.close()
        tmp[lbl] = t.name

    try:
        with st.status("Running audit pipeline…", expanded=True) as status_box:

            # Step 1 — Extraction
            st.write("**Step 1 / 3** — Extracting documents…")
            extracted = {}
            for lbl, path in tmp.items():
                r = extract_clinical_data(path)
                if r["status"] not in ("success",):
                    st.error(f"Extraction failed for {lbl}: {r.get('error')}")
                    st.session_state["stage"] = "failed"
                    _detach(); st.stop()
                extracted[lbl] = r
            st.session_state["extracted"] = extracted
            st.write(f"   ✅ {len(extracted)} documents extracted")

            # Step 2 — Translation
            st.write("**Step 2 / 3** — Translating & normalising…")
            merged_text   = "\n\n".join(f"=== {k.upper()} ===\n{v['text']}" for k,v in extracted.items())
            merged_tables = sum([v.get("tables",[]) for v in extracted.values()], [])
            tr = translate_and_normalize({"text":merged_text,"tables":merged_tables,"status":"success"})
            if tr.get("status") != "success":
                st.error(f"Translation failed: {tr.get('error')}")
                st.session_state["stage"] = "failed"
                _detach(); st.stop()
            st.session_state["translation"] = tr
            nm = tr.get("normalized_output", {})
            st.write(f"   ✅ Patient: **{nm.get('patient_name','(not found)')}** "
                     f"| lang: `{tr.get('detected_language','?')}` "
                     f"| confidence: `{tr.get('translation_confidence',0):.0%}`")

            # Step 3 — LangGraph pipeline
            st.write("**Step 3 / 3** — Running clinical validation agents…")
            init = {
                "discharge_path":tmp["discharge"],"lab_path":tmp["lab"],"bill_path":tmp["bill"],
                "extracted_discharge":extracted["discharge"],"extracted_lab":extracted["lab"],
                "extracted_bill":extracted["bill"],"normalized":nm,
                "completeness":{},"ehr_validation":{},"report":{},"rag_indexed":False,
                "current_stage":"starting","status":"running","error":None,
                "risk_level":None,"hitl_corrections":None,
            }
            result = discharge_pipeline.invoke(init, config=_thread_config())

            for p in tmp.values():
                try: os.unlink(p)
                except: pass

            # ── Detect interrupt vs completion ─────────────────
            # get_state().next is non-empty when the graph is paused at interrupt()
            graph_snapshot  = discharge_pipeline.get_state(_thread_config())
            is_interrupted  = bool(graph_snapshot.next)
            hitl_payload    = {}

            if is_interrupted:
                # Extract the payload passed to interrupt() inside completeness_node
                all_interrupts = [
                    intr
                    for task in graph_snapshot.tasks
                    for intr in task.interrupts
                ]
                hitl_payload = all_interrupts[0].value if all_interrupts else {}
                log.info(f"[App] Graph interrupted — payload keys: {list(hitl_payload.keys())}")

                # Patch pipeline_state so completeness expander has data to show
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
                status_box.update(label="⚠️ Review required — see below", state="error")
                st.warning("Pipeline paused — some clinical data is incomplete. "
                           "See the **Human Review** section below.")
            elif result.get("status") == "success":
                st.session_state["stage"] = "done"
                status_box.update(label="✅ Audit complete", state="complete")
                st.success(f"Complete — risk: **{result.get('risk_level','?').upper()}**")
            else:
                st.session_state["stage"] = "failed"
                status_box.update(label="❌ Failed", state="error")
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


# ═════════════════════════════════════════════════════════════
#  SECTION 2 — Inline step results
# ═════════════════════════════════════════════════════════════
extracted   = st.session_state.get("extracted")
translation = st.session_state.get("translation")
ps          = st.session_state.get("pipeline_state")

if extracted or translation or ps:
    st.markdown("---")
    st.markdown("### 📊 Pipeline Results")

if extracted:
    with st.expander("📄 Step 1 — Document Extraction", expanded=False):
        cols = st.columns(len(extracted))
        for i, (lbl, r) in enumerate(extracted.items()):
            with cols[i]:
                st.markdown(f"**{lbl.title()}**")
                st.markdown('<span class="badge badge-ok">✓ Success</span>', unsafe_allow_html=True)
                st.caption(f"`{r.get('file_name','—')}`")
                st.caption(f"{len(r.get('text',''))} chars · {len(r.get('tables',[]))} tables")
                if r.get("text"):
                    st.text_area("", value=r["text"][:300].replace("\n"," "),
                                 height=70, disabled=True, key=f"prev_{lbl}",
                                 label_visibility="collapsed")

if translation:
    nm   = translation.get("normalized_output", {})
    lang = translation.get("detected_language", "?")
    conf = translation.get("translation_confidence", 0)

    with st.expander("🌐 Step 2 — Translation & Normalisation", expanded=False):
        m1, m2, m3 = st.columns(3)
        m1.metric("Language", lang.upper())
        m2.metric("Confidence", f"{conf:.0%}")
        m3.metric("Medications found", len(nm.get("medications",[])))
        st.markdown("<br>", unsafe_allow_html=True)

        left, right = st.columns(2)
        with left:
            st.markdown("**Extracted fields**")
            rows = ""
            for f in ALL_PATIENT_FIELDS:
                val = nm.get(f)
                ok  = bool(val) or val == 0
                cls = "tick" if ok else "cross"
                sym = "✓"   if ok else "✗"
                disp = (str(val)[:50]+"…" if len(str(val))>50 else str(val)) if ok else "—"
                rows += (f'<div class="field-row"><span class="{cls}">{sym}</span>'
                         f'<span style="min-width:170px;color:#374151">{FIELD_LABELS[f]}</span>'
                         f'<span style="color:#64748b;font-size:.8rem">{disp}</span></div>')
            st.markdown(f'<div style="max-height:300px;overflow-y:auto">{rows}</div>',
                        unsafe_allow_html=True)
        with right:
            st.markdown("**Medications**")
            meds = nm.get("medications", [])
            if not meds:
                st.caption("None extracted.")
            for i, med in enumerate(meds, 1):
                st.markdown(
                    f'<div class="result-card" style="padding:8px 12px">'
                    f'<strong>{i}. {med.get("medicine_name","?")}</strong><br>'
                    f'<span style="color:#64748b;font-size:.82rem">'
                    f'Dose: {med.get("dosage","—")} | Freq: {med.get("frequency","—")} '
                    f'| Route: {med.get("route","—")}</span></div>',
                    unsafe_allow_html=True
                )

if ps:
    comp   = ps.get("completeness", {})
    ehr    = ps.get("ehr_validation", {})
    report = ps.get("report", {})

    # Show completeness whenever we have data OR are in HITL (patched state)
    hitl_payload_for_display = st.session_state.get("hitl_payload", {})
    comp_to_show = comp if comp else (
        {
            "passed"         : False,
            "missing_fields" : hitl_payload_for_display.get("missing_fields", []),
            "rule_violations": hitl_payload_for_display.get("rule_violations", []),
        }
        if hitl_payload_for_display else None
    )

    if comp_to_show is not None:
        passed  = comp_to_show.get("passed", False)
        missing = comp_to_show.get("missing_fields", [])
        viols   = comp_to_show.get("rule_violations", [])

        with st.expander("✅ Step 3 — Completeness Check",
                         expanded=(st.session_state["stage"] == "hitl")):
            badge = ('<span class="badge badge-ok">✓ PASSED</span>' if passed
                     else f'<span class="badge badge-warn">⚠ {len(missing)} missing · '
                          f'{len(viols)} violation(s)</span>')
            st.markdown(badge, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            if missing or viols:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Missing**")
                    for f in missing:
                        st.markdown(f'<div class="field-row"><span class="cross">✗</span>'
                                    f'<span style="color:#dc2626">{FIELD_LABELS.get(f,f)}</span></div>',
                                    unsafe_allow_html=True)
                    for v in viols:
                        st.markdown(f'<div class="field-row"><span class="cross">!</span>'
                                    f'<span style="color:#f97316;font-size:.82rem">{v}</span></div>',
                                    unsafe_allow_html=True)
                with c2:
                    st.markdown("**Present**")
                    for f in [f for f in ALL_PATIENT_FIELDS if f not in missing]:
                        st.markdown(f'<div class="field-row"><span class="tick">✓</span>'
                                    f'<span style="color:#374151">{FIELD_LABELS.get(f,f)}</span></div>',
                                    unsafe_allow_html=True)

    if st.session_state["stage"] == "hitl" and not ehr:
        with st.expander("🏥 Step 4 — EHR Validation", expanded=False):
            st.markdown('<span class="badge badge-neutral">⏳ Pending — runs after human review</span>',
                        unsafe_allow_html=True)
            st.caption("EHR validation will run automatically once you submit or skip the review below.")

    if ehr:
        disc   = ehr.get("discrepancies", [])
        passed = ehr.get("passed", False)
        with st.expander("🏥 Step 4 — EHR Validation", expanded=False):
            badge = ('<span class="badge badge-ok">✓ PASSED</span>' if passed
                     else f'<span class="badge badge-danger">✗ {len(disc)} discrepancy(ies)</span>')
            st.markdown(badge, unsafe_allow_html=True)
            if disc:
                st.markdown("<br>", unsafe_allow_html=True)
                for d in disc:
                    st.markdown(
                        f'<div class="result-card" style="border-left:3px solid #ef4444">'
                        f'<span class="badge badge-danger">{d.get("rule","—")}</span>'
                        f'&nbsp; {d.get("detail", str(d))}</div>',
                        unsafe_allow_html=True
                    )
            for c in ehr.get("allergy_conflicts", []):
                st.error(f"⚠️ Allergy conflict: {c}")

    if report and ps.get("status") == "success":
        risk  = report.get("risk_level", "unknown")
        score = report.get("risk_score", 0)
        rec   = report.get("recommendation", "—")
        rc    = {"low":"risk-low","medium":"risk-medium","high":"risk-high"}.get(risk,"risk-low")
        icon  = {"low":"🟢","medium":"🟡","high":"🔴"}.get(risk,"⬜")

        with st.expander("📋 Step 5 — Audit Report", expanded=True):
            st.markdown(
                f'<div class="{rc}"><strong style="font-size:1.05rem">'
                f'{icon} Risk: {risk.upper()}</strong><br>'
                f'<span style="color:#475569">Score: {score} &nbsp;|&nbsp; {rec}</span></div>',
                unsafe_allow_html=True
            )
            st.markdown("<br>", unsafe_allow_html=True)

            t1, t2, t3 = st.tabs(["Summary", "Risk Flags", "Full JSON"])
            with t1:
                st.markdown(
                    f'<pre style="white-space:pre-wrap;font-family:inherit;font-size:.88rem;'
                    f'background:#f8fafc;padding:16px;border-radius:8px">'
                    f'{report.get("summary","—")}</pre>',
                    unsafe_allow_html=True
                )
            with t2:
                flags = report.get("flags", [])
                if not flags:
                    st.caption("No risk flags.")
                for flag in sorted(flags, key=lambda x: -x.get("weight", 0)):
                    w    = flag.get("weight", 0)
                    kind = "danger" if w>=8 else ("warn" if w>=3 else "info")
                    bc   = {"danger":"#ef4444","warn":"#eab308","info":"#3b82f6"}[kind]
                    st.markdown(
                        f'<div class="result-card" style="border-left:3px solid {bc}">'
                        f'<span class="badge badge-{kind}">{flag.get("rule","—")}</span>'
                        f' <strong>+{w}</strong>'
                        f' <span style="color:#64748b;font-size:.88rem">{flag.get("detail","—")}</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
            with t3:
                st.json(report)

            st.divider()
            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                st.download_button(
                    "⬇ Download Report (JSON)",
                    data=json.dumps(report, indent=2, default=str),
                    file_name=f"audit_{ps.get('normalized',{}).get('patient_id','unknown')}.json",
                    mime="application/json",
                    use_container_width=True,
                )
            with dl_col2:
                pdf_bytes = generate_pdf_report(report)
                st.download_button(
                    "⬇ Download Report (PDF)",
                    data=pdf_bytes,
                    file_name=f"audit_{ps.get('normalized',{}).get('patient_id','unknown')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )


# ═════════════════════════════════════════════════════════════
#  SECTION 3 — HITL Review (inline, optional)
# ═════════════════════════════════════════════════════════════
if st.session_state["stage"] == "hitl":
    st.markdown("---")
    payload  = st.session_state.get("hitl_payload", {})
    missing  = payload.get("missing_fields", [])
    viols    = payload.get("rule_violations", [])
    nm       = payload.get("normalized",
                           st.session_state.get("pipeline_state", {}).get("normalized", {}))

    st.markdown("### ✏️ Human Review")
    st.markdown(
        '<div class="hitl-box"><strong>⚠️ Some clinical data could not be extracted.</strong><br>'
        '<span style="color:#78350f">Fill in what you know — any fields left blank are fine. '
        'Click <strong>Continue Pipeline</strong> or <strong>Skip</strong> to proceed.</span></div>',
        unsafe_allow_html=True
    )

    with st.form("hitl_form"):
        corrections = {}

        # Demographics
        demo = [f for f in ["patient_id","patient_name","age","gender","address"] if f in missing]
        if demo:
            st.markdown("#### 👤 Demographics")
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

        # Admission
        admit = [f for f in ["admission_date","discharge_date","ward","bed_no"] if f in missing]
        if admit:
            st.markdown("#### 🏥 Admission")
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

        # Clinical team
        team = [f for f in ["attending_physician","consulting_doctors"] if f in missing]
        if team:
            st.markdown("#### 👨‍⚕️ Clinical Team")
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

        # Allergies
        if "allergies" in missing:
            st.markdown("#### 💊 Allergies")
            raw = st.text_input("Allergies (comma-separated, or 'None')",
                                placeholder="Penicillin, Aspirin — or None")
            if raw.strip():
                corrections["allergies"] = ([] if raw.strip().lower()=="none"
                                            else [x.strip() for x in raw.split(",") if x.strip()])

        # Clinical notes
        notes_f = [f for f in ["discharge_diagnosis","follow_up_appointment","discharge_instructions"] if f in missing]
        if notes_f:
            st.markdown("#### 📋 Clinical Notes")
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

        # Prescription table
        if viols:
            st.markdown("#### 💊 Prescription Details")
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
            submit  = st.form_submit_button("▶ Continue Pipeline", type="primary")
        with col2:
            skip_it = st.form_submit_button("⏭ Skip — use as-is")

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

            # Same interrupt detection as initial run
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


# ═════════════════════════════════════════════════════════════
#  SECTION 4 — Pipeline logs (collapsed)
# ═════════════════════════════════════════════════════════════
if st.session_state.get("pipeline_logs"):
    st.markdown("---")
    with st.expander("🔍 Pipeline Logs", expanded=(stage == "failed")):
        st.code("\n".join(st.session_state["pipeline_logs"]), language="text")

# ═════════════════════════════════════════════════════════════
#  SECTION 5 — RAG Q&A
#  Shown whenever the FAISS index has data on disk — not gated
#  on the current session having run a pipeline.
# ═════════════════════════════════════════════════════════════
_INDEX_PATH = ROOT / "Data" / "faiss_index" / "index.faiss"

if _INDEX_PATH.exists():
    from agents.rag_agents.indexing_agent import list_indexed_patients
    from agents.rag_agents.rag_graph import rag_app

    st.markdown("---")
    st.markdown("### 💬 Ask about Patient Records")

    # ── Patient selector ──────────────────────────────────────
    patients = list_indexed_patients()
    current_patient_id = ps.get("normalized", {}).get("patient_id") if ps else None

    options_map = {"🌐 All patients": None}
    for pid, info in patients.items():
        label = f"{info.get('patient_name', pid)} ({pid})"
        options_map[label] = pid

    option_labels = list(options_map.keys())

    # Default to the current session's patient when available
    default_idx = 0
    if current_patient_id:
        for i, lbl in enumerate(option_labels):
            if options_map[lbl] == current_patient_id:
                default_idx = i
                break

    selected_label = st.selectbox(
        "Scope queries to:",
        option_labels,
        index=default_idx,
        key="rag_patient_selector",
    )
    selected_patient_id = options_map[selected_label]

    # Reset chat history whenever the patient scope changes
    if st.session_state.get("qa_patient_scope") != selected_patient_id:
        st.session_state["qa_history"]       = []
        st.session_state["qa_patient_scope"] = selected_patient_id

    if selected_patient_id:
        st.caption(f"Querying records scoped to **{selected_label}**.")
    else:
        st.caption("Querying across **all indexed patient records**.")

    # ── Chat history ──────────────────────────────────────────
    for q, a in st.session_state.get("qa_history", []):
        st.chat_message("user").write(q)
        st.chat_message("assistant").write(a)

    question = st.chat_input("Ask a question about patient records…")
    if question:
        st.chat_message("user").write(question)

        with st.spinner("Searching records…"):
            init_state = {
                "question"   : question,
                "patient_id" : selected_patient_id,   # None → search all patients
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
