"""
agents/translation_normalization_agent.py
─────────────────────────────────────────────────────────────────
Clinical Linguist Agent — Medical Lang Bridge Tool

Input  : extracted payload from extractor_agent
         {"text": str, "tables": [...], "status": "success", ...}

Output :
    {
        "status"            : "success" | "failed" | "empty",
        "normalized_output" : { ...structured patient dict... },
        "detected_language" : str,
        "translation_confidence" : float
    }

normalized_output always has these keys (empty string if not found):
    patient_id, patient_name, age, gender, address,
    admission_date, discharge_date, ward, bed_no,
    attending_physician, consulting_doctors,
    discharge_diagnosis, allergies,
    follow_up_appointment, discharge_instructions,
    medications: [{sl_no, medicine_name, strength, dosage,
                   frequency, route, period, remarks, total_quantity}],
    bill_paid, total_bill, discharge_ok, service_line,
    translation_confidence
"""

import json
import logging
from typing import Dict, Any

from dotenv import load_dotenv
import litellm

from configs.config import LITELLM_MODEL, ABBREVIATION_MAP, get_prompt

load_dotenv()
log = logging.getLogger("TranslationAgent")

_ABBR_TEXT = "\n".join(f"  {k} → {v}" for k, v in ABBREVIATION_MAP.items())
_SYSTEM_PROMPT = get_prompt(
    "translation_normalization",
    abbreviation_map=_ABBR_TEXT,
    output_schema=get_prompt("translation_normalization", "output_schema"),
)


# ═══════════════════════════════════════════════════════════════
#  Public entry point
# ═══════════════════════════════════════════════════════════════

def translate_and_normalize(extracted_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Args:
        extracted_payload : output dict from extract_clinical_data()

    Returns normalized output dict — see module docstring.
    """
    raw_text   = extracted_payload.get("text", "")
    raw_tables = extracted_payload.get("tables", [])

    if not raw_text.strip() and not raw_tables:
        log.warning("[TranslationAgent] Empty payload received")
        return {
            "status"                 : "empty",
            "normalized_output"      : _empty_normalized(),
            "detected_language"      : "unknown",
            "translation_confidence" : 0.0,
        }

    log.info("[TranslationAgent] Calling LiteLLM → Gemini")

    user_message = (
        f"CLINICAL DOCUMENT TEXT:\n{raw_text}\n\n"
        f"TABLES:\n{json.dumps(raw_tables, indent=2)}"
    )

    try:
        response = litellm.completion(
            model    = LITELLM_MODEL,
            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            temperature = 0.0,
            max_tokens  = 4096,
        )

        raw_output = response.choices[0].message.content.strip()

        # Strip markdown fences if model adds them
        if raw_output.startswith("```"):
            raw_output = raw_output.split("```")[1]
            if raw_output.startswith("json"):
                raw_output = raw_output[4:]
        raw_output = raw_output.strip()

        normalized = json.loads(raw_output)

        # Guarantee all required keys exist
        normalized = _fill_defaults(normalized)

        confidence = float(normalized.get("translation_confidence", 0.95))
        language   = normalized.get("detected_language", "en")

        log.info(
            f"[TranslationAgent] Done — language: {language}, "
            f"confidence: {confidence:.2f}"
        )

        return {
            "status"                 : "success",
            "normalized_output"      : normalized,
            "detected_language"      : language,
            "translation_confidence" : confidence,
        }

    except json.JSONDecodeError as e:
        log.error(f"[TranslationAgent] JSON parse failed: {e}\nRaw: {raw_output[:300]}")
        return {
            "status"                 : "failed",
            "normalized_output"      : _empty_normalized(),
            "detected_language"      : "unknown",
            "translation_confidence" : 0.0,
            "error"                  : f"JSON parse error: {e}",
        }
    except Exception as e:
        log.error(f"[TranslationAgent] LLM call failed: {e}")
        return {
            "status"                 : "failed",
            "normalized_output"      : _empty_normalized(),
            "detected_language"      : "unknown",
            "translation_confidence" : 0.0,
            "error"                  : str(e),
        }


# ═══════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════

def _empty_normalized() -> dict:
    return {
        "patient_id"             : "",
        "patient_name"           : "",
        "age"                    : "",
        "gender"                 : "",
        "address"                : "",
        "admission_date"         : "",
        "discharge_date"         : "",
        "ward"                   : "",
        "bed_no"                 : "",
        "attending_physician"    : "",
        "consulting_doctors"     : [],
        "discharge_diagnosis"    : "",
        "allergies"              : [],
        "follow_up_appointment"  : "",
        "discharge_instructions" : "",
        "service_line"           : "",
        "bill_paid"              : False,
        "total_bill"             : "",
        "discharge_ok"           : False,
        "detected_language"      : "unknown",
        "translation_confidence" : 0.0,
        "medications"            : [],
    }


def _fill_defaults(data: dict) -> dict:
    """Ensure all required keys exist — never let downstream agents KeyError."""
    defaults = _empty_normalized()
    for key, default_val in defaults.items():
        if key not in data or data[key] is None:
            data[key] = default_val
    # Ensure medications is always a list
    if not isinstance(data.get("medications"), list):
        data["medications"] = []
    return data