from src.state import AgentState
from langchain_core.messages import HumanMessage


def human_agent(state: AgentState) -> AgentState:
    print("==========YOUR TURN============")
    judge_messages = state["judge_messages"]
    cmp_messages = state["cmp_messages"]

    if state["human_hiding_place"] == "":
        hiding_place = input("Where do you want to hide your penny? ")
        judge_messages.append(
            HumanMessage(
                content=f"HUMAN: I have hidden my penny here: {hiding_place}\n"
            )
        )

        return {
            **state,
            "human_hiding_place": hiding_place,
            "judge_messages": judge_messages,
            "cmp_messages": cmp_messages,
        }
    else:
        # game has already begun ;)
        action = f"HUMAN: {input("What will be your next action? ")}\n"

        action_message = HumanMessage(content=action)
        judge_messages.append(action_message)
        cmp_messages.append(action_message)

        return {
            **state,
            "judge_messages": judge_messages,
            "cmp_messages": cmp_messages,
        }
