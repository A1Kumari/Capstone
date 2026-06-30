from langgraph.graph import StateGraph, END
from agents.rag_agents.rag_state import RAGState
from agents.rag_agents.retrieval_agent import retrieve_node
from agents.rag_agents.augmentation_agent import augmentation_grader_node
from agents.rag_agents.generation_agent import generation_node
from agents.rag_agents.reflection_agent import reflection_node

def route_reflection(state: RAGState) -> str:
   is_grounded = state.get("is_grounded", False)
   loop_count = state.get("loop_count", 0)
   
   if is_grounded:
       return "useful"
   elif loop_count >= 3:
       print("--- MAX LOOPS REACHED: Exiting ---")
       return "max_loops"
   else:
       print("--- RETRYING GENERATION ---")
       return "not_grounded"

workflow = StateGraph(RAGState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade_documents", augmentation_grader_node)
workflow.add_node("generate", generation_node)
workflow.add_node("reflect", reflection_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_edge("grade_documents", "generate")
workflow.add_edge("generate", "reflect")

workflow.add_conditional_edges(
   "reflect",
   route_reflection,
   {
       "useful": END,
       "not_grounded": "retrieve",  # re-retrieve fresh docs, not just re-generate
       "max_loops": END
   }
)

rag_app = workflow.compile()