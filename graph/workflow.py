from langgraph.graph import StateGraph, START, END
from graph.nodes.retrieve import retrieve
from graph.nodes.reason import reason
from graph.state import AgentState
from graph.nodes.verify import has_math_equations, verify_math_node
from typing import Literal


def route_after_reason(state: AgentState) -> Literal["verify", "end"]:
    reasoning_for_detection = state.reasoning_logs or state.raw_output or ""
    if has_math_equations(reasoning_for_detection):
        return "verify"
    return "end"

def create_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("reason", reason)
    workflow.add_node("verify", verify_math_node)


    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "reason")
    workflow.add_conditional_edges(
        "reason",
        route_after_reason,
        {"verify": "verify", "end": END},
    )
    workflow.add_edge("verify", END)
    compiled_graph = workflow.compile()
    return compiled_graph

if __name__ == "__main__":
    app = create_graph()
    print("Graph compiled successfully.")
    png_data = app.get_graph().draw_mermaid_png()
    output_path = "graph/workflow_graph.png"
    
    with open(output_path, "wb") as f:
        f.write(png_data)
        
    print(f"Graph visualization saved to: {output_path}")
