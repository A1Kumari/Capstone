import hashlib
import json
import logging
from datetime import datetime
from html import escape
from pathlib import Path

import yaml

log = logging.getLogger("ReportingAgent")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RULES_PATH   = _PROJECT_ROOT / "configs" / "rules.yaml"

def _load_rules() -> dict:
    with open(_RULES_PATH, "r") as f:
        return yaml.safe_load(f)

def _rules_sha256() -> str:
    return hashlib.sha256(_RULES_PATH.read_bytes()).hexdigest()

RULES = _load_rules()
WEIGHTS    = RULES["risk_scoring_matrix"]["weights"]
THRESHOLDS = RULES["risk_scoring_matrix"]["thresholds"]
HARD_GUARDRAILS = set(RULES["risk_scoring_matrix"]["hitl_hard_guardrails"])
RECOMMENDATIONS = RULES["reporting"]["recommendations"]
QUALITY     = RULES["quality_thresholds"]
BIZ_RULES   = RULES["business_rules"]
HIGH_RISK_MEDS = set(RULES["clinical_validation_policies"]["high_risk_meds_need_counseling"])

_REPORTING_CFG  = RULES["reporting"]
_OUTPUT_DIR     = (_PROJECT_ROOT / _REPORTING_CFG.get("output_dir", "data/reports")).resolve()
_REPORT_FORMATS = set(_REPORTING_CFG.get("formats", ["json", "html"]))


def generate_report(
    normalized: dict,
    completeness: dict,
    ehr_validation: dict,
) -> dict:
    audit_trail = []
    flags       = []          # list of {rule, weight, detail}
    score       = 0
    hard_blocks = []

    def _flag(rule: str, detail: str, override_weight: int = None):
        weight = override_weight if override_weight is not None else WEIGHTS.get(rule, 0)
        flags.append({"rule": rule, "weight": weight, "detail": detail})
        nonlocal score
        score += weight
        if rule in HARD_GUARDRAILS:
            hard_blocks.append(detail)
        _audit(f"FLAG [{rule}] +{weight} — {detail}")

    def _audit(message: str):
        audit_trail.append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "agent"    : "ReportingAgent",
            "message"  : message,
        })

    _audit("Reporting agent started")

    # 1. Completeness gaps
    _audit("Evaluating completeness results")

    missing_fields = completeness.get("missing_fields", [])
    for field in missing_fields:
        if field == "address":
            _flag("missing_address", f"Address missing")
        elif field == "gender":
            _flag("missing_gender", f"Gender missing")
        else:
            _flag("missing_mandatory_field", f"Mandatory field missing: {field}")

    for violation in completeness.get("rule_violations", []):
        _flag("incomplete_prescription_fields", violation)

    # 2. EHR discrepancies
    _audit("Evaluating EHR validation results")

    for disc in ehr_validation.get("discrepancies", []):
        rule   = disc.get("rule", "medication_omission")
        detail = disc.get("detail", str(disc))
        _flag(rule, detail)

    # 3. Allergy contradictions
    allergy_conflicts = ehr_validation.get("allergy_conflicts", [])
    for conflict in allergy_conflicts:
        _flag("allergy_contradiction", f"Allergy conflict: {conflict}")

    # 4. High-risk med missing from EHR history
    meds_in_discharge = _get_med_names(normalized)
    ehr_med_history   = set(ehr_validation.get("ehr_medications", []))
    for med in meds_in_discharge:
        if med in HIGH_RISK_MEDS and med not in ehr_med_history:
            _flag("high_risk_med_missing_in_ehr",
                  f"High-risk med {med} in discharge but not in EHR history")

    # 5. High-risk med with no counseling note
    counseling_noted = set(ehr_validation.get("counseling_noted_for", []))
    for med in meds_in_discharge:
        if med in HIGH_RISK_MEDS and med not in counseling_noted:
            _flag("high_risk_med_no_counseling",
                  f"No counseling note found for high-risk med: {med}")

    # 6. Translation confidence
    trans_conf = normalized.get("translation_confidence")
    if trans_conf is not None and trans_conf < QUALITY["translation_confidence_min"]:
        _flag("low_translation_confidence",
              f"Translation confidence {trans_conf:.2f} below threshold "
              f"{QUALITY['translation_confidence_min']}")

    # 7. Missing follow-up (skip if EHR agent already flagged it to avoid double-counting)
    ehr_flagged_rules = {d.get("rule") for d in ehr_validation.get("discrepancies", [])}
    if not normalized.get("follow_up_appointment") and "followup_missing" not in ehr_flagged_rules:
        _flag("followup_missing", "Follow-up appointment not documented")

    # 8. Abnormal labs without follow-up
    for lab in ehr_validation.get("abnormal_labs_unresolved", []):
        _flag("abnormal_lab_unresolved",
              f"Abnormal lab result with no documented action: {lab}")

    # 9. Bill unpaid with discharge approved
    bill_paid    = normalized.get("bill_paid", False)
    discharge_ok = normalized.get("discharge_ok", False)
    if BIZ_RULES["bill_must_be_paid_before_release"] and discharge_ok and not bill_paid:
        _flag("bill_unpaid_with_discharge_ok",
              "Bill outstanding but discharge marked as approved — release blocked")

    # 10. Service-line hard guardrails
    service_line = normalized.get("service_line", "").lower()
    if "pediatric" in service_line:
        _flag("service_line_pediatric", "Pediatric case — always HITL")
    if "obstetric" in service_line or "maternity" in service_line:
        _flag("service_line_obstetric", "Obstetric case — always HITL")
    if "oncology" in service_line or "cancer" in service_line:
        _flag("service_line_oncology", "Oncology case — always HITL")

    hard_blocked = len(hard_blocks) > 0

    if hard_blocked or score > THRESHOLDS["medium_max"]:
        risk_level = "high"
    elif score > THRESHOLDS["low_max"]:
        risk_level = "medium"
    else:
        risk_level = "low"

    recommendation = RECOMMENDATIONS[risk_level]

    _audit(f"Risk score: {score} | Risk level: {risk_level.upper()} | Hard blocked: {hard_blocked}")
    _audit(f"Recommendation: {recommendation}")

    summary = _build_summary(normalized, risk_level, recommendation)
    _audit("Patient-friendly discharge summary generated")

    report = {
        "status"             : "success",
        "risk_level"         : risk_level,
        "risk_score"         : score,
        "recommendation"     : recommendation,
        "flags"              : flags,
        "hard_blocked"       : hard_blocked,
        "hard_block_reasons" : hard_blocks,
        "summary"            : summary,
        "audit_trail"        : audit_trail,
        "rules_version"      : _rules_sha256(),
        "generated_at"       : datetime.now().isoformat(timespec="seconds"),
        # pass-through for Streamlit display
        "patient_id"         : normalized.get("patient_id"),
        "patient_name"       : normalized.get("patient_name"),
        "discharge_diagnosis": normalized.get("discharge_diagnosis"),
        "bill_status"        : "Paid" if normalized.get("bill_paid") else "Unpaid",
        "total_bill"         : normalized.get("total_bill"),
    }

    log.info(
        f"[Reporting] {normalized.get('patient_id')} | "
        f"score={score} | level={risk_level} | flags={len(flags)}"
    )
    return report


def _build_summary(normalized: dict, risk_level: str, recommendation: str) -> str:
    abbr_map = RULES.get("normalization_standards", {}).get("abbreviation_map", {})

    def _expand(text: str) -> str:
        if not text:
            return ""
        for abbr, full in abbr_map.items():
            text = text.replace(abbr, full)
        return text

    name      = normalized.get("patient_name", "Patient")
    age       = normalized.get("age", "—")
    gender    = normalized.get("gender", "—")
    pid       = normalized.get("patient_id", "—")
    admitted  = normalized.get("admission_date", "—")
    discharged= normalized.get("discharge_date", "—")
    ward      = normalized.get("ward", "—")
    doctor    = normalized.get("attending_physician", "—")
    diagnosis = _expand(normalized.get("discharge_diagnosis", "—"))
    followup  = normalized.get("follow_up_appointment", "Not scheduled")
    instructions = _expand(normalized.get("discharge_instructions", "None provided"))
    bill_status  = "Paid" if normalized.get("bill_paid") else "Outstanding"
    total_bill   = normalized.get("total_bill", "—")
    release_status = "Released" if normalized.get("discharge_ok") else "Not approved for release"

    meds = normalized.get("medications", [])
    med_lines = []
    for i, med in enumerate(meds, 1):
        freq    = _expand(med.get("frequency", "—"))
        remarks = _expand(med.get("remarks", ""))
        med_lines.append(
            f"  {i}. {med.get('medicine_name','—')} {med.get('strength','')} — "
            f"{med.get('dosage','—')} {freq} via {med.get('route','—')} "
            f"for {med.get('period','—')}. {remarks}"
        )
    med_text = "\n".join(med_lines) if med_lines else "  No medications recorded."

    lab_vendor  = normalized.get("lab_vendor") or "Not documented"
    labs        = normalized.get("lab_results", [])
    lab_lines = []
    for lab in labs:
        unit  = lab.get("unit", "")
        rng   = lab.get("reference_range", "")
        extra = f" {unit}".rstrip() + (f" (ref: {rng})" if rng else "")
        lab_lines.append(
            f"  - {lab.get('name','—')}: {lab.get('value','—')}{extra} "
            f"[{lab.get('status','—').upper()}]"
        )
    lab_text = "\n".join(lab_lines) if lab_lines else "  No lab results recorded."

    summary = f"""
PATIENT DISCHARGE SUMMARY
══════════════════════════════════════════════════════════

Patient        : {name}  |  Age: {age}  |  Gender: {gender}
Patient ID     : {pid}
Ward / Bed     : {ward}
Admitted       : {admitted}
Discharged     : {discharged}
Doctor         : {doctor}

──────────────────────────────────────────────────────────
DISCHARGE DIAGNOSIS
──────────────────────────────────────────────────────────
{diagnosis}

──────────────────────────────────────────────────────────
MEDICINES TO TAKE AT HOME
──────────────────────────────────────────────────────────
{med_text}

──────────────────────────────────────────────────────────
DISCHARGE INSTRUCTIONS
──────────────────────────────────────────────────────────
{instructions}

──────────────────────────────────────────────────────────
FOLLOW-UP APPOINTMENT
──────────────────────────────────────────────────────────
{followup}

──────────────────────────────────────────────────────────
LAB REPORTS  (Vendor: {lab_vendor})
──────────────────────────────────────────────────────────
{lab_text}

──────────────────────────────────────────────────────────
BILL SUMMARY
──────────────────────────────────────────────────────────
Total Bill     : {total_bill}
Payment Status : {bill_status}

──────────────────────────────────────────────────────────
AUDIT OUTCOME
──────────────────────────────────────────────────────────
Risk Level         : {risk_level.upper()}
Decision           : {recommendation}
Discharge Release  : {release_status}
══════════════════════════════════════════════════════════
    """.strip()

    return summary


def _get_med_names(normalized: dict) -> list[str]:
    meds = normalized.get("medications", [])
    return [m.get("medicine_name", "") for m in meds if m.get("medicine_name")]


def persist_report(report: dict) -> dict:
    """Write the audit report to disk per rules.yaml's `reporting` config
    (output_dir / formats). Non-fatal — persistence failures are logged
    but never raised, so a disk/permission issue can't fail the pipeline.
    """
    written = {}
    try:
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        patient_id = report.get("patient_id") or "unknown"
        stamp = (report.get("generated_at") or datetime.now().isoformat(timespec="seconds")).replace(":", "-")
        stem  = f"{patient_id}_{stamp}"

        if "json" in _REPORT_FORMATS:
            json_path = _OUTPUT_DIR / f"{stem}.json"
            json_path.write_text(json.dumps(report, indent=2, default=str))
            written["json"] = str(json_path)
            log.info(f"[Reporting] JSON report written: {json_path}")

        if "html" in _REPORT_FORMATS:
            html_path = _OUTPUT_DIR / f"{stem}.html"
            html_path.write_text(_render_html_report(report))
            written["html"] = str(html_path)
            log.info(f"[Reporting] HTML report written: {html_path}")

    except Exception as exc:
        log.warning(f"[Reporting] Failed to persist report to disk (non-fatal): {exc}")

    return written


def _render_html_report(report: dict) -> str:
    risk = (report.get("risk_level") or "unknown").upper()
    colors = {
        "LOW":    ("#dcfce7", "#15803d"),
        "MEDIUM": ("#fef3c7", "#92400e"),
        "HIGH":   ("#fee2e2", "#b91c1c"),
    }
    bg, fg = colors.get(risk, ("#f0f0f0", "#3c3c3c"))

    flags = sorted(report.get("flags") or [], key=lambda f: -f.get("weight", 0))
    flag_rows = "".join(
        f'<tr style="background:{"#fee2e2" if f.get("weight", 0) >= 8 else ("#fef3c7" if f.get("weight", 0) >= 3 else "#fff")}">'
        f'<td>{escape(str(f.get("rule", "")))}</td>'
        f'<td>+{f.get("weight", 0)}</td>'
        f'<td>{escape(str(f.get("detail", "")))}</td></tr>'
        for f in flags
    ) or '<tr><td colspan="3">No flags raised.</td></tr>'

    audit_rows = "".join(
        f'<tr><td>{escape(str(a.get("timestamp", "")))}</td>'
        f'<td>{escape(str(a.get("agent", "")))}</td>'
        f'<td>{escape(str(a.get("message", "")))}</td></tr>'
        for a in (report.get("audit_trail") or [])
    ) or '<tr><td colspan="3">No audit entries.</td></tr>'

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Discharge Audit Report — {escape(str(report.get('patient_id', '')))}</title>
<style>
  body {{ font-family: -apple-system, Helvetica, Arial, sans-serif; max-width: 860px; margin: 32px auto; color:#222; padding: 0 16px; }}
  h1 {{ font-size: 1.4rem; }}
  h2 {{ font-size: 1.05rem; margin-top: 28px; border-bottom: 2px solid #eee; padding-bottom: 4px; }}
  .risk-banner {{ background:{bg}; color:{fg}; padding:14px 20px; border-radius:8px; font-weight:700; margin:16px 0; }}
  .meta {{ color:#666; font-size:.85rem; margin-bottom:20px; }}
  pre.summary {{ background:#f7f7f7; padding:16px; border-radius:8px; white-space:pre-wrap; font-size:.9rem; line-height:1.5; }}
  table {{ width:100%; border-collapse:collapse; margin:12px 0 24px; font-size:.85rem; }}
  th, td {{ border:1px solid #ddd; padding:6px 10px; text-align:left; vertical-align:top; }}
  th {{ background:#f0f0f0; }}
</style></head>
<body>
  <h1>Clinical Discharge Audit Report</h1>
  <div class="meta">Generated: {escape(str(report.get('generated_at', '—')))}
    &middot; Rules version: {escape(str(report.get('rules_version', '—')))}</div>
  <div class="risk-banner">RISK: {risk} &nbsp;|&nbsp; Score: {report.get('risk_score', 0)}
    &nbsp;|&nbsp; {escape(str(report.get('recommendation', '—')))}</div>

  <h2>Discharge Summary</h2>
  <pre class="summary">{escape(str(report.get('summary', '')))}</pre>

  <h2>Audit Flags ({len(flags)})</h2>
  <table><thead><tr><th>Rule</th><th>Weight</th><th>Detail</th></tr></thead>
  <tbody>{flag_rows}</tbody></table>

  <h2>Audit Trail</h2>
  <table><thead><tr><th>Timestamp</th><th>Agent</th><th>Message</th></tr></thead>
  <tbody>{audit_rows}</tbody></table>
</body></html>"""