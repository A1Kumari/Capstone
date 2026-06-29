# retrieval_agent.py
from agents.rag_agents.rag_state import RAGState
from agents.rag_agents.indexing_agent import get_retriever

def retrieve_node(state: RAGState) -> dict:
    """
    Retrieves documents relevant to the question.
    Falls back to unfiltered search if patient-scoped retrieval returns nothing.
    """
    print("--- RETRIEVAL AGENT: Fetching Documents ---")
    question = state["question"]
    loop_count = state.get("loop_count", 0)
    patient_id = state.get("patient_id")

    retriever = get_retriever(patient_id)
    documents = retriever.invoke(question)

    if not documents and patient_id:
        print("--- RETRIEVAL AGENT: No patient-scoped results, falling back to global search ---")
        retriever = get_retriever(None)
        documents = retriever.invoke(question)

    return {"documents": documents, "question": question, "loop_count": loop_count}