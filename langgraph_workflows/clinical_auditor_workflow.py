import sys
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

from pipeline_state import PipelineState
from agents.completeness_agent   import check_clinical_completeness
from agents.ehr_validation_agent import validate_discharge
from agents.reporting_agent      import generate_report
from agents.rag_agents.indexing_agent import index_documents

log = logging.getLogger("Workflow")


# ── Nodes ─────────────────────────────────────────────────────

def completeness_node(state: PipelineState) -> dict:
    log.info("[completeness] START")
    result = check_clinical_completeness(state["normalized"])

    if not result.get("passed"):
        log.warning(f"[completeness] incomplete — missing: {result['missing_fields']}, "
                    f"violations: {len(result['rule_violations'])}")

        corrections = interrupt({
            "missing_fields"  : result["missing_fields"],
            "rule_violations" : result["rule_violations"],
            "normalized"      : state["normalized"],
        })

        normalized = {**state["normalized"], **(corrections or {})}
        log.info(f"[completeness] resumed — corrections applied: {bool(corrections)}")

        return {
            "normalized"     : normalized,
            "completeness"   : result,
            "hitl_corrections": corrections or {},
            "current_stage"  : "completeness",
            "status"         : "running",
        }

    log.info("[completeness] passed")
    return {
        "completeness" : result,
        "current_stage": "completeness",
        "status"       : "running",
    }


def ehr_node(state: PipelineState) -> dict:
    log.info("[ehr] START")

    normalized = state["normalized"]
    if state.get("hitl_corrections"):
        normalized = {**normalized, **state["hitl_corrections"]}

    result = validate_discharge(normalized)
    passed = result.get("passed", True)

    if not passed:
        log.warning(f"[ehr] {len(result.get('discrepancies', []))} discrepancy(ies)")
    else:
        log.info("[ehr] passed")

    return {
        "ehr_validation": result,
        "current_stage" : "ehr_validation",
        "status"        : "running",
    }


def reporting_node(state: PipelineState) -> dict:
    log.info("[reporting] START")
    try:
        report = generate_report(state["normalized"], state["completeness"], state["ehr_validation"])
        log.info(f"[reporting] risk={report.get('risk_level')} score={report.get('risk_score')}")
        return {
            "report"       : report,
            "risk_level"   : report.get("risk_level"),
            "current_stage": "reporting",
            "status"       : "running",
        }
    except Exception as exc:
        log.exception("[reporting] FAILED")
        return {"status": "failed", "error": str(exc), "current_stage": "reporting"}


def rag_index_node(state: PipelineState) -> dict:
    log.info("[rag_index] START")
    try:
        index_documents(normalized=state["normalized"], report=state["report"])
        log.info("[rag_index] indexed")
        return {"rag_indexed": True, "current_stage": "rag_indexed", "status": "success"}
    except Exception as exc:
        log.warning(f"[rag_index] failed (non-fatal): {exc}")
        return {"rag_indexed": False, "current_stage": "rag_indexed", "status": "success"}


def route_after_reporting(state: PipelineState) -> str:
    return "end" if state["status"] == "failed" else "rag_index"


# ── Graph ─────────────────────────────────────────────────────

def _build() -> StateGraph:
    g = StateGraph(PipelineState)

    g.add_node("completeness", completeness_node)
    g.add_node("ehr",          ehr_node)
    g.add_node("reporting",    reporting_node)
    g.add_node("rag_index",    rag_index_node)

    g.set_entry_point("completeness")
    g.add_edge("completeness", "ehr")
    g.add_edge("ehr",          "reporting")
    g.add_conditional_edges("reporting", route_after_reporting,
                            {"rag_index": "rag_index", "end": END})
    g.add_edge("rag_index", END)

    return g


discharge_pipeline = _build().compile(checkpointer=MemorySaver())
