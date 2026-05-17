from langgraph.graph import END
from agents.multi_agent.states import SupervisorState


def coordinator_edge(state: SupervisorState) -> str:
    if state["coordinator_final"]:
        return END
    nxt = state["coordinator_next"]
    if nxt == "troubleshooting":
        return "troubleshooting"
    if nxt == "summarizer":
        return "summarizer"
    return END
