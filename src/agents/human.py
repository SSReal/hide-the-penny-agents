from src.state import AgentState
from langchain_core.messages import HumanMessage


def human_agent(state: AgentState) -> AgentState:
    print("==========YOUR TURN============")
    if state["human_hiding_place"] == "":
        hiding_place = input("Where do you want to hide your penny? ")
        state["human_hiding_place"] = hiding_place
        state["history"] += "HUMAN: I have hidden my penny!\n\n"
    else:
        # the game has already begun ;)
        action = f"HUMAN: {input("What will be your next action? ")}\n\n"
        state["history"] += action

    print("")
    return state
