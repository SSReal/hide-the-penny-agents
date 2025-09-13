from src.agents.judge import judge
from src.agents.computer import cmp_agent
from src.agents.human import human_agent
from src.state import AgentState
from langgraph.graph import StateGraph, END


def turn_router(state: AgentState) -> str:
    if state["turn"] == 0:
        return "cmp"
    elif state["turn"] == 1:
        return "human"
    else:
        return "end"


stateGraph = (
    StateGraph(AgentState)
    .add_node("judge", judge)
    .add_node("human_agent", human_agent)
    .add_node("cmp_agent", cmp_agent)
    .set_entry_point("judge")
    .add_conditional_edges(
        "judge", turn_router, {"cmp": "cmp_agent", "human": "human_agent", "end": END}
    )
    .add_edge("cmp_agent", "judge")
    .add_edge("human_agent", "judge")
)
