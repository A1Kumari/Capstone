# rag_state.py
from typing import TypedDict, List
from langchain_core.documents import Document

class RAGState(TypedDict):
   """
   Represents the state of our RAG workflow.
   """
   question: str
   patient_id: str
   documents: List[Document]
   generation: str
   is_grounded: bool
   loop_count: int
   scores: dict