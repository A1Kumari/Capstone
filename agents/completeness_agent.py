"""
agents/completeness_agent.py
─────────────────────────────────────────────────────────────────
Clinical Data Completeness Agent — Clinical Rules Engine Tool

Input  : normalized dict  (state["normalized"] from pipeline_state)
         i.e. the direct output of _fill_defaults() in translation agent
         NOT the full translation payload wrapper

Output :
    {
        "passed"          : bool,         # False = HITL required
        "missing_fields"  : [str],        # mandatory field names missing
        "rule_violations" : [str],        # prescription row gaps, policy hits
        "validated_data"  : dict,         # echo back normalized for convenience
    }

The graph checks result.get("passed") to decide routing.
The reporting agent reads result["missing_fields"] and result["rule_violations"].
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, List

import yaml

log = logging.getLogger("CompletenessAgent")

# ── Load rules.yaml ────────────────────────────────────────────
_RULES_PATH = Path(__file__).resolve().parent.parent / "configs" / "rules.yaml"

def _load_rules() -> dict:
    try:
        with open(_RULES_PATH) as f:
            return yaml.safe_load(f)
    except Exception as e:
        log.error(f"[Completeness] Cannot load rules.yaml: {e} — using defaults")
        return {
            "mandatory_clinical_fields"   : ["patient_id", "patient_name", "discharge_diagnosis"],
            "mandatory_prescription_fields": ["medicine_name", "frequency", "total_quantity"],
            "clinical_validation_policies" : {},
        }

RULES = _load_rules()


# ═══════════════════════════════════════════════════════════════
#  Public entry point
# ═══════════════════════════════════════════════════════════════

def check_clinical_completeness(normalized: Dict[str, Any]) -> Dict[str, Any]:
    """
    Args:
        normalized : the normalized_output dict directly
                     (state["normalized"] in the LangGraph state)

    Returns completeness result dict — see module docstring.
    """
    log.info("[Completeness] Running validation rules check")

    missing_fields  : List[str] = []
    rule_violations : List[str] = []

    # ── 1. Mandatory clinical fields ──────────────────────────
    for field in RULES.get("mandatory_clinical_fields", []):
        value = normalized.get(field)
        # Empty string, None, empty list all count as missing
        if not value and value != 0:
            missing_fields.append(field)
            log.debug(f"[Completeness] Missing: {field}")

    # ── 2. Prescription row validation ───────────────────────
    medications = normalized.get("medications", [])

    if not medications:
        # medications list itself is missing — already caught above
        # but add a specific violation for clarity
        rule_violations.append("No medications/prescriptions found in document")
    else:
        rx_fields = RULES.get("mandatory_prescription_fields", [])
        for idx, med in enumerate(medications, start=1):
            for rx_field in rx_fields:
                val = med.get(rx_field)
                if not val and val != 0:
                    rule_violations.append(
                        f"Prescription row {idx} ({med.get('medicine_name', '?')})"
                        f": missing '{rx_field}'"
                    )

    # ── 3. Clinical policy checks ─────────────────────────────
    policies = RULES.get("clinical_validation_policies", {})

    # Allergy field present but empty — flag it
    if policies.get("allergy_must_not_match_prescription"):
        allergies    = [a.lower() for a in normalized.get("allergies", [])]
        med_names    = [
            m.get("medicine_name", "").lower()
            for m in medications
        ]
        for med in med_names:
            if med and med in allergies:
                rule_violations.append(
                    f"Allergy contradiction: '{med}' is both prescribed and listed as an allergy"
                )

    # Abnormal lab follow-up (policy flag only — EHR agent does full check)
    if policies.get("abnormal_lab_requires_followup"):
        if not normalized.get("follow_up_appointment"):
            rule_violations.append(
                "Policy: abnormal_lab_requires_followup is ON but no follow-up appointment found"
            )

    # ── 4. Compute result ─────────────────────────────────────
    passed = len(missing_fields) == 0 and len(rule_violations) == 0

    if passed:
        log.info("[Completeness] All checks passed")
    else:
        log.warning(
            f"[Completeness] HITL required — "
            f"missing: {missing_fields}, violations: {len(rule_violations)}"
        )

    return {
        "passed"         : passed,
        "missing_fields" : missing_fields,
        "rule_violations": rule_violations,
        "validated_data" : normalized,     # pass-through for convenience
    }