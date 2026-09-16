from langgraph.graph import StateGraph, START, END
from core.state import CopilotState
from core.nodes import orchestrator_node, developer_node

def build_graph():
    """
    Wires the agents together using a directed graph.
    """
    # 1. Initialize the builder with our shared Memory (State schema)
    builder = StateGraph(CopilotState)
    
    # 2. Register the worker nodes
    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("developer", developer_node)
    
    # 3. Define the routing logic (conditional edge)
    def route_after_orchestrator(state: CopilotState):
        """
        Checks the state to decide the next move.
        If the OS agent successfully found a file path, route to Developer.
        Otherwise, we are finished.
        """
        if state.get("local_file_paths") and len(state["local_file_paths"]) > 0:
            return "developer"
        
        return END

    # 4. Wire the graph logic
    # Set the entry point so the orchestrator always runs first
    builder.add_edge(START, "orchestrator")
    
    # Use the conditional router to decide what happens next
    builder.add_conditional_edges("orchestrator", route_after_orchestrator)
    
    # If the developer runs, it will always go to the end afterward
    builder.add_edge("developer", END)
    
    # 5. Compile the graph into an executable application
    return builder.compile()