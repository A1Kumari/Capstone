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
from agents.translation_normalization_agent import translate_and_normalize
from langgraph_workflows.clinical_auditor_workflow import discharge_pipeline


class ClinicalFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return

        file_path = event.src_path
        path = Path(file_path)
        log.info("=" * 60)
        log.info(f"[Monitor] New file detected: {path.name}")
        log.info("=" * 60)

        try:
            # ── Step 1: Extract ──────────────────────────────────
            log.info(f"[Monitor] STEP 1 — Extracting: {path.name}")
            extracted = extract_clinical_data(file_path)
            log.info(f"[Monitor] Extraction status: {extracted.get('status')}")
            log.debug(f"[Monitor] Extracted text length: {len(extracted.get('text', ''))}")
            log.debug(f"[Monitor] Extracted tables count: {len(extracted.get('tables', []))}")

            if extracted.get("status") != "success":
                log.error(f"[Monitor] Extraction FAILED: {extracted.get('error')}")
                return

            # ── Step 2: Translate ────────────────────────────────
            log.info("[Monitor] STEP 2 — Translating & normalising…")
            translation = translate_and_normalize(extracted)
            log.info(f"[Monitor] Translation status: {translation.get('status')}")
            log.info(f"[Monitor] Detected language: {translation.get('detected_language')}")
            log.info(f"[Monitor] Translation confidence: {translation.get('translation_confidence')}")

            if translation.get("status") != "success":
                log.error(f"[Monitor] Translation FAILED: {translation.get('error')}")
                return

            normalized = translation.get("normalized_output", {})
            log.info(f"[Monitor] Patient: {normalized.get('patient_name')} | ID: {normalized.get('patient_id')}")
            log.info(f"[Monitor] Medications found: {len(normalized.get('medications', []))}")
            log.info(f"[Monitor] Diagnosis: {normalized.get('discharge_diagnosis')}")
            log.debug(f"[Monitor] Full normalized keys: {list(normalized.keys())}")

            # ── Step 3: LangGraph pipeline ───────────────────────
            log.info("[Monitor] STEP 3 — Invoking LangGraph pipeline…")
            thread_id = str(uuid.uuid4())
            config    = {"configurable": {"thread_id": thread_id}}

            initial_state = {
                "discharge_path"      : file_path,
                "lab_path"            : "",
                "bill_path"           : "",
                "extracted_discharge" : extracted,
                "extracted_lab"       : {},
                "extracted_bill"      : {},
                "normalized"          : normalized,
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

            # ── Step 4: Result ───────────────────────────────────
            log.info("[Monitor] STEP 4 — Pipeline complete")
            log.info(f"[Monitor] Status    : {final_state.get('status')}")
            log.info(f"[Monitor] Stage     : {final_state.get('current_stage')}")
            log.info(f"[Monitor] Risk level: {final_state.get('risk_level')}")

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
