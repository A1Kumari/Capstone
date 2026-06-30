import json
import litellm
from agents.rag_agents.rag_state import RAGState

from dotenv import load_dotenv
from configs.config import LITELLM_MODEL, RAG_UNKNOWN_RESPONSE, get_prompt

load_dotenv()

def reflection_node(state: RAGState) -> dict:
    """
    Evaluates the RAG Triad (Context Relevance, Groundedness, Answer Relevance).
    """
    print("--- REFLECTION AGENT: Evaluating RAG Triad ---")
    documents = state.get("documents", [])
    generation = state["generation"]
    question = state["question"]
    loop_count = state.get("loop_count", 0) + 1

    if not documents and RAG_UNKNOWN_RESPONSE in generation:
         return {
             "is_grounded": True,
             "loop_count": loop_count,
             "scores": {"context_relevance": 0.0, "groundedness": 1.0, "answer_relevance": 1.0}
         }

    system_prompt = get_prompt("rag_reflection")
    
    context_str = "\n\n".join([doc.page_content for doc in documents])
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"User Question: {question} \n\n Retrieved Facts: \n\n {context_str} \n\n LLM Generation: {generation}"}
    ]
    
    try:
        response = litellm.completion(
            model=LITELLM_MODEL,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0,
        )
        content = response.choices[0].message.content
        scores = json.loads(content)
        
        context_rel = float(scores.get("context_relevance", 0.0))
        groundedness = float(scores.get("groundedness", 0.0))
        answer_rel = float(scores.get("answer_relevance", 0.0))
        
        is_grounded = groundedness >= 0.75
    except Exception as e:
        print(f"--- REFLECTION: Error parsing triad check: {e} ---")
        is_grounded = False
        context_rel, groundedness, answer_rel = 0.0, 0.0, 0.0

    if is_grounded:
        print(f"--- REFLECTION: Answer is Grounded (Score: {groundedness}) ---")
    else:
        print(f"--- REFLECTION: Hallucination Detected! (Score: {groundedness}) ---")
        
    return {
        "is_grounded": is_grounded, 
        "loop_count": loop_count,
        "scores": {
            "context_relevance": context_rel,
            "groundedness": groundedness,
            "answer_relevance": answer_rel
        }
    }