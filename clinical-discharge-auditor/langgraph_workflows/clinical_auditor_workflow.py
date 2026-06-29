"""
pipeline_state.py
─────────────────────────────────────────────────────────────────
Single source of truth for the LangGraph pipeline state.
Every agent node reads from and writes to this TypedDict.

Placed at project root so both agents/ and langgraph_workflows/
can import it without circular deps:

    from pipeline_state import PipelineState
"""

from typing import TypedDict, Optional, Literal


class PipelineState(TypedDict):

    # ── Inputs (set once by Streamlit before graph runs) ─────────
    # Absolute paths to the 3 uploaded files
    discharge_path : str
    lab_path       : str
    bill_path      : str

    # ── Stage 2 — Extractor output ───────────────────────────────
    # Already done before graph starts; passed in as pre-filled state
    extracted_discharge : dict   # {text, tables, is_scanned, ...}
    extracted_lab       : dict
    extracted_bill      : dict

    # ── Stage 3 — Translation / Normalization output ─────────────
    # Already done before graph starts; passed in as pre-filled state
    normalized : dict            # merged, English, normalized payload
                                 # keys your completeness agent already expects

    # ── Stage 4 — Completeness output ────────────────────────────
    completeness : dict
    # {
    #   "missing_fields"  : [...],
    #   "rule_violations" : [...],
    #   "passed"          : bool
    # }

    # ── Stage 5 — EHR Validation output ──────────────────────────
    ehr_validation : dict
    # {
    #   "discrepancies" : [...],
    #   "passed"        : bool
    # }

    # ── Stage 6 — Reporting output ───────────────────────────────
    report : dict
    # {
    #   "risk_level"      : "low" | "medium" | "high",
    #   "risk_score"      : int,
    #   "recommendation"  : str,
    #   "audit_trail"     : [...],
    #   "summary"         : str
    # }

    # ── RAG ───────────────────────────────────────────────────────
    rag_indexed : bool           # True once indexing_agent is done

    # ── Pipeline control ─────────────────────────────────────────
    current_stage : str          # for Streamlit progress display
    status        : Literal["running", "hitl_required", "success", "failed"]
    error         : Optional[str]

    # ── HITL ──────────────────────────────────────────────────────
    hitl_required   : bool
    hitl_reason     : Optional[str]   # which stage triggered it + why
    hitl_corrections: Optional[dict]  # clinician fills this on Streamlit
                                      # graph re-invoked with this populated

    # ── Final output ──────────────────────────────────────────────
    risk_level : Optional[Literal["low", "medium", "high"]]