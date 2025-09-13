from src.state import AgentState
from langchain_core.messages import ChatMessage, HumanMessage, AIMessage
from src.prompts.computer import cmp_system_prompt, hide_prompt_text
from src.utils import print_llm_stream, get_tag
from src.llm import llm


def cmp_agent(state: AgentState) -> AgentState:
    print("==========COMPUTER TURN============")

    system_prompt = ChatMessage(role="system", content=cmp_system_prompt)
    curr_messages = [system_prompt]

    judge_messages = state["judge_messages"]
    cmp_messages = state["cmp_messages"]

    if state["cmp_hiding_place"] == "":
        # hasn't hidden the penny yet
        hiding_prompt = ChatMessage(
            role="system",
            content=hide_prompt_text,
        )

        response_stream = llm.stream(curr_messages + cmp_messages + [hiding_prompt])
        response = print_llm_stream(
            response_stream, print_reasoning=False, print_response=False
        )

        cmp_hiding_place = get_tag(str(response), "CMP_HIDING_PLACE")
        response_hiding_place = (
            f"COMPUTER: I have hidden my penny here: {cmp_hiding_place}\n"
        )

        judge_messages.append(HumanMessage(content=response_hiding_place))
        cmp_messages.append(AIMessage(content=response_hiding_place))

        return {
            **state,
            "cmp_hiding_place": cmp_hiding_place,
            "judge_messages": judge_messages,
            "cmp_messages": cmp_messages,
        }
    else:
        # game has already begun ;)
        action_response_stream = llm.stream(curr_messages + cmp_messages)
        action_response = print_llm_stream(
            action_response_stream, print_reasoning=False
        )

        judge_messages.append(HumanMessage(content=action_response))
        cmp_messages.append(AIMessage(content=action_response))

        return {
            **state,
            "judge_messages": judge_messages,
            "cmp_messages": cmp_messages,
        }
