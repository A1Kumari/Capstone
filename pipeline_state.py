from typing import TypedDict, Optional, Literal


class PipelineState(TypedDict):
    # ── Pre-graph inputs ──────────────────────────────────────
    discharge_path      : str
    lab_path            : str
    bill_path           : str

    extracted_discharge : dict
    extracted_lab       : dict
    extracted_bill      : dict
    normalized          : dict

    # ── Node outputs ──────────────────────────────────────────
    completeness        : dict
    ehr_validation      : dict
    report              : dict
    rag_indexed         : bool

    # ── Control ───────────────────────────────────────────────
    current_stage       : str
    status              : Literal["running", "success", "failed"]
    error               : Optional[str]
    risk_level          : Optional[Literal["low", "medium", "high"]]

    # ── HITL (informational — routing handled by interrupt()) ─
    hitl_corrections    : Optional[dict]
