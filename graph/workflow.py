from langgraph.graph import StateGraph, START, END
from graph.nodes.retrieve import retrieve
from graph.nodes.reason import reason
from graph.state import AgentState
from graph.verify import verify_math_node

def create_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("reason", reason)
    workflow.add_node("verify", verify_math_node)


    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "reason")
    workflow.add_edge("reason", "verify")
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