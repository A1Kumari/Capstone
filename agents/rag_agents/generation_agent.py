import litellm
from agents.rag_agents.rag_state import RAGState

from dotenv import load_dotenv
from configs.config import LITELLM_MODEL, LLM_BASE_URL, FOUNDATION_MODEL_API_KEY, RAG_UNKNOWN_RESPONSE, get_prompt

load_dotenv()

def generation_node(state: RAGState) -> dict:
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
           api_key=FOUNDATION_MODEL_API_KEY,
           api_base=LLM_BASE_URL,
           messages=messages,
       )
       generation = response.choices[0].message.content
   except Exception as e:
       print(f"--- GENERATION AGENT: LLM call failed: {e} ---")
       generation = RAG_UNKNOWN_RESPONSE

   return {"generation": generation}