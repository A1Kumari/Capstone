import sys
import uuid
import logging
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger("Monitor")

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import json

from agents.extractor_agent import extract_clinical_data
from langgraph_workflows.clinical_auditor_workflow import discharge_pipeline


# Filename convention for grouping a discharge report with its matching lab
# report / bill in the watched folder, e.g.:
#   P1008_discharge.pdf, P1008_lab.pdf, P1008_bill.pdf
# Everything before the first underscore is the case/patient ID; the doc type
# is whichever of these keywords appears in the filename.
def _classify_doc_type(filename: str) -> str:
    name = filename.lower()
    if "bill" in name:
        return "bill"
    if "lab" in name:
        return "lab"
    if "discharge" in name:
        return "discharge"
    return "unknown"


def _case_prefix(filename: str) -> str:
    stem = Path(filename).stem
    return stem.split("_", 1)[0] if "_" in stem else stem


def _find_sibling(folder: Path, prefix: str, doc_type: str, exclude_name: str) -> str:
    """Find a file in `folder` matching the same case prefix and doc type."""
    candidates = sorted(
        p for p in folder.iterdir()
        if p.is_file()
        and p.name != exclude_name
        and _case_prefix(p.name) == prefix
        and _classify_doc_type(p.name) == doc_type
    )
    if len(candidates) > 1:
        log.warning(
            f"[Monitor] Multiple {doc_type} candidates for case '{prefix}': "
            f"{[c.name for c in candidates]} — using {candidates[0].name}"
        )
    return str(candidates[0]) if candidates else ""


class ClinicalFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return

        file_path = event.src_path
        path = Path(file_path)
        doc_type = _classify_doc_type(path.name)

        # Lab/bill documents don't trigger the pipeline on their own — they
        # wait to be picked up when their matching discharge report lands.
        if doc_type in ("lab", "bill"):
            log.info(f"[Monitor] {doc_type.title()} document detected: {path.name} "
                     f"— waiting to be paired with a discharge report")
            return

        log.info("=" * 60)
        log.info(f"[Monitor] New discharge document detected: {path.name}")
        log.info("=" * 60)

        try:
            log.info(f"[Monitor] STEP 1 — Extracting: {path.name}")
            extracted = extract_clinical_data(file_path)
            log.info(f"[Monitor] Extraction status: {extracted.get('status')}")
            log.debug(f"[Monitor] Extracted text length: {len(extracted.get('text', ''))}")
            log.debug(f"[Monitor] Extracted tables count: {len(extracted.get('tables', []))}")

            if extracted.get("status") != "success":
                log.error(f"[Monitor] Extraction FAILED: {extracted.get('error')}")
                return

            # ── Find and extract matching lab report / bill, if present ──
            case_prefix   = _case_prefix(path.name)
            folder        = path.parent
            lab_file      = _find_sibling(folder, case_prefix, "lab",  path.name)
            bill_file     = _find_sibling(folder, case_prefix, "bill", path.name)

            docs = {"discharge": extracted}
            lab_path_str  = ""
            bill_path_str = ""

            if lab_file:
                log.info(f"[Monitor] Found matching lab report: {Path(lab_file).name}")
                el = extract_clinical_data(lab_file)
                if el.get("status") == "success":
                    docs["lab"] = el
                    lab_path_str = lab_file
                else:
                    log.warning(f"[Monitor] Lab extraction failed, continuing without it: {el.get('error')}")

            if bill_file:
                log.info(f"[Monitor] Found matching bill: {Path(bill_file).name}")
                eb = extract_clinical_data(bill_file)
                if eb.get("status") == "success":
                    docs["bill"] = eb
                    bill_path_str = bill_file
                else:
                    log.warning(f"[Monitor] Bill extraction failed, continuing without it: {eb.get('error')}")

            if not lab_file and not bill_file:
                log.info(f"[Monitor] No matching lab/bill documents found for case "
                         f"'{case_prefix}' — proceeding with discharge report only")

            merged_text   = "\n\n".join(f"=== {k.upper()} ===\n{v.get('text','')}" for k, v in docs.items())
            merged_tables = sum([v.get("tables", []) for v in docs.values()], [])
            merged_discharge = {"text": merged_text, "tables": merged_tables, "status": "success"}

            log.info("[Monitor] STEP 2 — Invoking LangGraph pipeline "
                     "(translation → completeness → EHR → reporting → RAG index)…")
            thread_id = str(uuid.uuid4())
            config    = {"configurable": {"thread_id": thread_id}}

            initial_state = {
                "discharge_path"      : file_path,
                "lab_path"            : lab_path_str,
                "bill_path"           : bill_path_str,
                "extracted_discharge" : merged_discharge,
                "extracted_lab"       : docs.get("lab", {}),
                "extracted_bill"      : docs.get("bill", {}),
                "normalized"          : {},
                "translation"         : {},
                "completeness"        : {},
                "ehr_validation"      : {},
                "report"              : {},
                "rag_indexed"         : False,
                "current_stage"       : "starting",
                "status"              : "running",
                "error"               : None,
                "hitl_corrections"    : None,
                "risk_level"          : None,
            }

            final_state = discharge_pipeline.invoke(initial_state, config=config)

            log.info("[Monitor] STEP 3 — Pipeline complete")
            log.info(f"[Monitor] Status    : {final_state.get('status')}")
            log.info(f"[Monitor] Stage     : {final_state.get('current_stage')}")
            log.info(f"[Monitor] Risk level: {final_state.get('risk_level')}")

            translation = final_state.get("translation", {})
            log.info(f"[Monitor] Translation status: {translation.get('status')}")
            log.info(f"[Monitor] Detected language: {translation.get('detected_language')}")
            log.info(f"[Monitor] Translation confidence: {translation.get('translation_confidence')}")

            normalized = final_state.get("normalized", {})
            log.info(f"[Monitor] Patient: {normalized.get('patient_name')} | ID: {normalized.get('patient_id')}")
            log.info(f"[Monitor] Medications found: {len(normalized.get('medications', []))}")
            log.info(f"[Monitor] Diagnosis: {normalized.get('discharge_diagnosis')}")

            # Detect HITL interrupt via graph checkpoint state
            graph_snapshot = discharge_pipeline.get_state(config)
            if graph_snapshot.next:
                completeness = final_state.get("completeness", {})
                log.warning("[Monitor] Pipeline paused — human review required (headless mode: skipping HITL)")
                log.warning(f"[Monitor] Missing fields   : {completeness.get('missing_fields', [])}")
                log.warning(f"[Monitor] Rule violations  : {completeness.get('rule_violations', [])}")

            if final_state.get("status") == "failed":
                log.error(f"[Monitor] Pipeline ERROR: {final_state.get('error')}")

            report = final_state.get("report", {})
            if report:
                log.info(f"[Monitor] Risk score: {report.get('risk_score')}")
                log.info(f"[Monitor] Recommendation: {report.get('recommendation')}")

        except Exception:
            log.error("[Monitor] UNCAUGHT EXCEPTION in on_created:")
            log.error(traceback.format_exc())


def start_monitoring():
    incoming_path = ROOT / "Data" / "incoming"
    incoming_path.mkdir(parents=True, exist_ok=True)

    handler  = ClinicalFileHandler()
    observer = Observer()
    observer.schedule(handler, str(incoming_path), recursive=False)
    observer.start()
    log.info(f"[Monitor] Watching: {incoming_path.resolve()}")
    log.info("[Monitor] Drop a file into that folder to trigger the pipeline.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("[Monitor] Shutting down…")
        observer.stop()
    observer.join()


if __name__ == "__main__":
    start_monitoring()
