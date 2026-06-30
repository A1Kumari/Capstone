import os
import json
from pathlib import Path
from datetime import datetime, timezone
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
from configs.config import EMBEDDING_MODEL, GEMINI_API_KEY

load_dotenv()

INDEX_DIR    = Path(__file__).resolve().parent.parent.parent / "Data" / "faiss_index"
PATIENTS_FILE = INDEX_DIR / "patients.json"


def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GEMINI_API_KEY,
    )


def list_indexed_patients() -> dict:
    if PATIENTS_FILE.exists():
        with open(PATIENTS_FILE) as f:
            return json.load(f)
    return {}


def _register_patient(patient_id: str, patient_name: str) -> None:
    registry = list_indexed_patients()
    registry[patient_id] = {
        "patient_id"  : patient_id,
        "patient_name": patient_name,
        "indexed_at"  : datetime.now(timezone.utc).isoformat(),
    }
    with open(PATIENTS_FILE, "w") as f:
        json.dump(registry, f, indent=2)


def get_retriever(patient_id=None):
    embeddings = get_embeddings()

    if INDEX_DIR.exists() and (INDEX_DIR / "index.faiss").exists():
        vectorstore = FAISS.load_local(
            str(INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    else:
        vectorstore = FAISS.from_texts(["No patient records found."], embedding=embeddings)

    search_kwargs = {"k": 6}
    if patient_id:
        search_kwargs["filter"] = {"patient_id": patient_id}

    return vectorstore.as_retriever(search_kwargs=search_kwargs)


def index_documents(normalized, report):
    print("--- INDEXING AGENT: Indexing patient data ---")
    patient_id   = normalized.get("patient_id", "unknown")
    patient_name = normalized.get("patient_name", "Unknown Patient")

    docs_to_index = [
        f"PATIENT REPORT SUMMARY:\n{report.get('summary', '')}",
        f"CLINICAL DATA:\n{json.dumps(normalized, indent=2)}",
    ]

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.create_documents(
        docs_to_index,
        metadatas=[{"patient_id": patient_id} for _ in docs_to_index],
    )

    embeddings = get_embeddings()
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    if (INDEX_DIR / "index.faiss").exists():
        vectorstore = FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)
        vectorstore.add_documents(chunks)
    else:
        vectorstore = FAISS.from_documents(chunks, embedding=embeddings)

    vectorstore.save_local(str(INDEX_DIR))
    _register_patient(patient_id, patient_name)
    print(f"--- INDEXING AGENT: Indexed {len(chunks)} chunks for {patient_id} ({patient_name}) ---")
