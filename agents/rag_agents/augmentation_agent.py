# augmentation_agent.py
import json
import litellm
from agents.rag_agents.rag_state import RAGState

from dotenv import load_dotenv
from configs.config import LITELLM_MODEL, get_prompt

load_dotenv()

def augmentation_grader_node(state: RAGState) -> dict:
   """
   Filters out irrelevant documents.
   """
   print("--- AUGMENTATION AGENT: Grading Document Relevance ---")
   question = state["question"]
   documents = state["documents"]

   system_prompt = get_prompt("rag_augmentation")
   
   filtered_docs = []
   for d in documents:
       messages = [
           {"role": "system", "content": system_prompt},
           {"role": "user", "content": f"Retrieved document: \n\n {d.page_content} \n\n User question: {question}"}
       ]
       
       try:
           response = litellm.completion(
               model=LITELLM_MODEL,
               messages=messages,
               response_format={"type": "json_object"},
               temperature=0,
           )
           
           content = response.choices[0].message.content
           score = json.loads(content).get("binary_score", "no")
           
           if score.lower() == "yes":
               print("--- GRADE: Document Relevant ---")
               filtered_docs.append(d)
           else:
               print("--- GRADE: Document Irrelevant ---")
       except Exception as e:
           print(f"--- GRADE: Error grading document, skipping. Error: {e} ---")
           
   return {"documents": filtered_docs}