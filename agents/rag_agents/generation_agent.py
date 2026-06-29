# generation_agent.py
import litellm
from agents.rag_agents.rag_state import RAGState

from dotenv import load_dotenv
from configs.config import LITELLM_MODEL, RAG_UNKNOWN_RESPONSE, get_prompt

load_dotenv()

def generation_node(state: RAGState) -> dict:
   """
   Generates an answer using the retrieved context.
   """
   print("--- GENERATION AGENT: Drafting Response ---")
   question = state["question"]
   documents = state["documents"]

   if not documents:
       return {"generation": RAG_UNKNOWN_RESPONSE}

   context_str = "\n\n".join([doc.page_content for doc in documents])
   system_prompt = get_prompt("rag_generation", context=context_str)

   messages = [
       {"role": "system", "content": system_prompt},
       {"role": "user", "content": question}
   ]

   try:
       response = litellm.completion(
           model=LITELLM_MODEL,
           messages=messages,
       )
       generation = response.choices[0].message.content
   except Exception as e:
       print(f"--- GENERATION AGENT: LLM call failed: {e} ---")
       generation = RAG_UNKNOWN_RESPONSE

   return {"generation": generation}