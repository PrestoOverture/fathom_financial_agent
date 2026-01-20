from langgraph.graph import StateGraph, START, END
from graph.nodes.retrieve import retrieve
from graph.nodes.reason import reason
from graph.state import AgentState

def create_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("reason", reason)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "reason")
    workflow.add_edge("reason", END)
    compiled_graph = workflow.compile()
    return compiled_graph