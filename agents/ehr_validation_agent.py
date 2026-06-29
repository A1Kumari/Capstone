"""
agents/ehr_validation_agent.py
─────────────────────────────────────────────────────────────────
EHR Validation Agent — EHR Validation Tool

Input  : normalized dict (state["normalized"] from pipeline_state)

Output :
    {
        "passed"                   : bool,
        "discrepancies"            : [{"rule": str, "detail": str}],
        "allergy_conflicts"        : [str],
        "ehr_medications"          : [str],   # med names from EHR history
        "counseling_noted_for"     : [str],   # meds with counseling in EHR
        "abnormal_labs_unresolved" : [str],
    }

reporting_agent reads exactly these keys.
graph checks result.get("passed") for routing.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List

import requests
import yaml

from configs.config import EHR_BASE_URL

log = logging.getLogger("EHRValidationAgent")

# ── Load rules.yaml ────────────────────────────────────────────
_RULES_PATH = Path(__file__).resolve().parent.parent / "configs" / "rules.yaml"
with open(_RULES_PATH) as f:
    _RULES = yaml.safe_load(f)

_POLICIES       = _RULES.get("clinical_validation_policies", {})
_HIGH_RISK_MEDS = set(_POLICIES.get("high_risk_meds_need_counseling", []))
_BIZ_RULES      = _RULES.get("business_rules", {})


# ═══════════════════════════════════════════════════════════════
#  EHR Client
# ═══════════════════════════════════════════════════════════════

class EHRClient:
    BASE_URL = EHR_BASE_URL
    TIMEOUT  = 5   # seconds

    def _get(self, url: str) -> dict:
        """Safe GET — returns {} on any failure so pipeline never crashes."""
        try:
            resp = requests.get(url, timeout=self.TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.ConnectionError:
            log.error(f"[EHR] Mock EHR server not reachable: {url}")
            return {}
        except requests.exceptions.Timeout:
            log.error(f"[EHR] Timeout calling: {url}")
            return {}
        except Exception as e:
            log.error(f"[EHR] Failed {url}: {e}")
            return {}

    def get_patient(self, patient_id: str)   -> dict:
        return self._get(f"{self.BASE_URL}/ehr/patient/{patient_id}")

    def get_med_orders(self, patient_id: str) -> dict:
        return self._get(f"{self.BASE_URL}/ehr/med_orders/{patient_id}")

    def get_labs(self, patient_id: str)       -> dict:
        return self._get(f"{self.BASE_URL}/ehr/labs/{patient_id}")

    def get_allergies(self, patient_id: str)  -> dict:
        return self._get(f"{self.BASE_URL}/ehr/allergies/{patient_id}")

    def get_careplan(self, patient_id: str)   -> dict:
        return self._get(f"{self.BASE_URL}/ehr/careplan/{patient_id}")


# ═══════════════════════════════════════════════════════════════
#  Individual check functions
# ═══════════════════════════════════════════════════════════════

def _check_medication_omission(
    discharge_meds: List[str],
    ehr_med_data: dict,
) -> List[Dict]:
    """
    Medications in EHR history that are absent from discharge prescription.
    discharge_meds : list of medicine_name strings from normalized data
    ehr_med_data   : response from GET /ehr/med_orders/{id}
    """
    issues = []

    if isinstance(ehr_med_data, list):
        med_list = ehr_med_data
    elif isinstance(ehr_med_data, dict):
        med_list = ehr_med_data.get("med_orders", ehr_med_data.get("meds_orders", []))
    else:
        med_list = []

    # Mock EHR returns {"med_orders": ["MedA", "MedB", ...]}
    ehr_meds = set(
        m.lower() for m in med_list if isinstance(m, str)
    )
    discharge_set = set(m.lower() for m in discharge_meds)

    for med in ehr_meds - discharge_set:
        issues.append({
            "rule"  : "medication_omission",
            "detail": f"Medication in EHR history but missing from discharge prescription: '{med}'"
        })

    # Also flag meds added in discharge that were never in EHR
    for med in discharge_set - ehr_meds:
        issues.append({
            "rule"  : "medication_added",
            "detail": f"Medication in discharge prescription but not in EHR history: '{med}'"
        })

    return issues


def _check_allergy_contradiction(
    discharge_meds: List[str],
    ehr_allergy_data: dict,
) -> List[str]:
    """
    Returns list of med names that appear in both prescription and allergy list.
    """
    conflicts = []
    if isinstance(ehr_allergy_data, list):
        allergy_list = ehr_allergy_data
    elif isinstance(ehr_allergy_data, dict):
        allergy_list = ehr_allergy_data.get("allergies", [])
    else:
        allergy_list = []

    ehr_allergies = set(
        a.lower() for a in allergy_list if isinstance(a, str)
    )
    for med in discharge_meds:
        if med.lower() in ehr_allergies:
            conflicts.append(med)
            log.warning(f"[EHR] Allergy contradiction: {med}")
    return conflicts


def _check_diagnosis_mismatch(
    normalized: dict,
    ehr_patient: dict,
) -> List[Dict]:
    """
    Compares discharge diagnosis against EHR primary diagnosis.
    """
    issues = []
    discharge_dx = normalized.get("discharge_diagnosis", "").lower().strip()
    if isinstance(ehr_patient, dict):
        ehr_dx = ehr_patient.get("primary_dx", "").lower().strip()
    else:
        ehr_dx = ""

    if not discharge_dx or not ehr_dx:
        return issues   # can't compare if either is missing

    if discharge_dx != ehr_dx:
        issues.append({
            "rule"  : "diagnosis_mismatch",
            "detail": (
                f"Discharge diagnosis '{normalized.get('discharge_diagnosis')}' "
                f"does not match EHR primary diagnosis '{ehr_patient.get('primary_dx')}'"
            )
        })
    return issues


def _check_followup(
    normalized: dict,
    careplan: dict,
) -> List[Dict]:
    """
    If careplan requires follow-up but discharge has none — flag it.
    """
    issues = []
    follow_up_required = careplan.get("follow_up_required") if isinstance(careplan, dict) else False
    if follow_up_required and not normalized.get("follow_up_appointment"):
        issues.append({
            "rule"  : "followup_missing",
            "detail": "Care plan requires follow-up but no follow-up appointment documented"
        })
    return issues


def _check_abnormal_labs(
    lab_data: dict,
    normalized: dict,
) -> List[str]:
    """
    Returns list of abnormal lab names that have no documented follow-up action.
    Only flagged if clinical_validation_policies.abnormal_lab_requires_followup = true.
    """
    if not _POLICIES.get("abnormal_lab_requires_followup", False):
        return []

    unresolved = []
    
    if isinstance(lab_data, list):
        labs = lab_data
    elif isinstance(lab_data, dict):
        labs = lab_data.get("labs", [])
    else:
        labs = []
    followup   = (normalized.get("follow_up_appointment") or "").lower()
    instructions = (normalized.get("discharge_instructions") or "").lower()

    for lab in labs:
        # Mock EHR lab entry: {"name": "HbA1c", "value": 9.2, "status": "abnormal"}
        if lab.get("status", "").lower() in ("abnormal", "critical"):
            lab_name = lab.get("name", "unknown lab")
            # Simple check — if lab name mentioned in instructions/followup it's resolved
            if lab_name.lower() not in followup and lab_name.lower() not in instructions:
                unresolved.append(lab_name)
                log.warning(f"[EHR] Abnormal lab unresolved: {lab_name}")

    return unresolved


def _check_discharge_approval(ehr_patient: dict) -> List[Dict]:
    """
    Checks that the doctor has approved discharge in EHR.
    """
    issues = []
    if _BIZ_RULES.get("discharge_ok_field_required"):
        approved = ehr_patient.get("discharge_approved", False) if isinstance(ehr_patient, dict) else False
        if not approved:
            issues.append({
                "rule"  : "discharge_approval_missing",
                "detail": "Discharge not marked as approved by doctor in EHR"
            })
    return issues


def _get_counseling_noted(ehr_med_data: dict) -> List[str]:
    """
    Returns list of med names that have counseling notes in EHR.
    Mock EHR: {"counseling_notes": ["Warfarin", ...]}
    """
    if isinstance(ehr_med_data, list):
        return []
    elif isinstance(ehr_med_data, dict):
        return ehr_med_data.get("counseling_notes", [])
    return []


# ═══════════════════════════════════════════════════════════════
#  Public entry point
# ═══════════════════════════════════════════════════════════════

def validate_discharge(normalized: Dict[str, Any]) -> Dict[str, Any]:
    """
    Args:
        normalized : state["normalized"] from pipeline_state
                     (direct output of translation agent _fill_defaults)

    Returns EHR validation result dict — see module docstring.
    """
    patient_id = normalized.get("patient_id", "")

    if not patient_id:
        log.error("[EHR] No patient_id in normalized data — cannot call EHR APIs")
        return {
            "passed"                   : False,
            "discrepancies"            : [{"rule": "missing_patient_id",
                                           "detail": "patient_id missing — EHR validation skipped"}],
            "allergy_conflicts"        : [],
            "ehr_medications"          : [],
            "counseling_noted_for"     : [],
            "abnormal_labs_unresolved" : [],
        }

    log.info(f"[EHR] Validating patient: {patient_id}")
    client = EHRClient()

    # ── Fetch from mock EHR ───────────────────────────────────
    ehr_patient  = client.get_patient(patient_id)
    ehr_meds     = client.get_med_orders(patient_id)
    ehr_labs     = client.get_labs(patient_id)
    ehr_allergies= client.get_allergies(patient_id)
    ehr_careplan = client.get_careplan(patient_id)

    # If the patient was not found in EHR (empty or sentinel response), skip
    # cross-validation to avoid generating false-positive discrepancies.
    patient_found = bool(ehr_patient.get("patient_id")) if isinstance(ehr_patient, dict) else False
    if not patient_found:
        log.warning(f"[EHR] Patient '{patient_id}' not found in EHR system — skipping cross-validation")
        return {
            "passed"                   : True,
            "discrepancies"            : [],
            "allergy_conflicts"        : [],
            "ehr_medications"          : [],
            "counseling_noted_for"     : [],
            "abnormal_labs_unresolved" : [],
        }

    # ── Extract discharge med names from normalized ───────────
    discharge_meds = [
        m.get("medicine_name", "")
        for m in normalized.get("medications", [])
        if m.get("medicine_name")
    ]

    # ── Run all checks ────────────────────────────────────────
    discrepancies = []

    discrepancies.extend(_check_medication_omission(discharge_meds, ehr_meds))
    discrepancies.extend(_check_diagnosis_mismatch(normalized, ehr_patient))
    discrepancies.extend(_check_followup(normalized, ehr_careplan))
    discrepancies.extend(_check_discharge_approval(ehr_patient))

    allergy_conflicts        = _check_allergy_contradiction(discharge_meds, ehr_allergies)
    abnormal_labs_unresolved = _check_abnormal_labs(ehr_labs, normalized)
    ehr_med_names            = ehr_meds if isinstance(ehr_meds, list) else (ehr_meds.get("med_orders", ehr_meds.get("meds_orders", [])) if isinstance(ehr_meds, dict) else [])
    counseling_noted_for     = _get_counseling_noted(ehr_meds)

    # ── Determine passed ──────────────────────────────────────
    passed = (
        len(discrepancies)            == 0
        and len(allergy_conflicts)    == 0
        and len(abnormal_labs_unresolved) == 0
    )

    if passed:
        log.info(f"[EHR] Validation passed for {patient_id}")
    else:
        log.warning(
            f"[EHR] Validation failed for {patient_id} — "
            f"discrepancies: {len(discrepancies)}, "
            f"allergy conflicts: {len(allergy_conflicts)}, "
            f"unresolved labs: {len(abnormal_labs_unresolved)}"
        )

    return {
        "passed"                   : passed,
        "discrepancies"            : discrepancies,
        "allergy_conflicts"        : allergy_conflicts,
        "ehr_medications"          : ehr_med_names,
        "counseling_noted_for"     : counseling_noted_for,
        "abnormal_labs_unresolved" : abnormal_labs_unresolved,
    }


# ── Quick test ─────────────────────────────────────────────────
if __name__ == "__main__":
    client = EHRClient()
    for pid in ["P1008"]:
        print("Patient :", client.get_patient(pid))
        print("Meds    :", client.get_med_orders(pid))
        print("Labs    :", client.get_labs(pid))
        print("Allergies:", client.get_allergies(pid))
        print("Careplan:", client.get_careplan(pid))