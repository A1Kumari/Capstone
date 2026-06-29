# retrieval_agent.py
from agents.rag_agents.rag_state import RAGState
from agents.rag_agents.indexing_agent import get_retriever

def retrieve_node(state: RAGState) -> dict:
   """
   Retrieves documents relevant to the question.
   """
   print("--- RETRIEVAL AGENT: Fetching Documents ---")
   question = state["question"]
   loop_count = state.get("loop_count", 0)
   
   retriever = get_retriever(state.get("patient_id"))
   documents = retriever.invoke(question)
   
   return {"documents": documents, "question": question, "loop_count": loop_count}