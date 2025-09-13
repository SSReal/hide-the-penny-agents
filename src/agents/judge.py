from src.state import AgentState
from langchain_core.messages import SystemMessage, ChatMessage, AIMessage
from src.prompts.judge import (
    judge_system_prompt,
    setup_prompt,
    both_hidden_prompt as judge_both_hidden_prompt,
    game_summary_prompt_text,
)
from src.prompts.computer import both_hidden_prompt as cmp_both_hidden_prompt
import random
from src.utils import print_llm_stream, get_tag, get_bool_tag
from src.llm import llm


def judge(state: AgentState) -> AgentState:
    print("==========JUDGE TURN============")

    system_prompt = ChatMessage(role="system", content=judge_system_prompt)
    curr_messages = [system_prompt]

    judge_messages = state["judge_messages"]
    cmp_messages = state["cmp_messages"]

    if len(state["judge_messages"]) == 0:
        # game hasn't started yet
        # set it up
        scene_setting_prompt = ChatMessage(
            role="system",
            content=setup_prompt,
        )
        response_stream = llm.stream(curr_messages + [scene_setting_prompt])
        response = print_llm_stream(response_stream, print_reasoning=False)

        scene_desc = get_tag(str(response), "SCENE")
        scene_str = f"JUDGE: The scene is set as follows: \n{scene_desc}\n"

        judge_messages.append(AIMessage(content=scene_str))
        cmp_messages.append(SystemMessage(content=scene_str))

        return {
            **state,
            "scene_desc": scene_desc,
            "judge_messages": judge_messages,
            "cmp_messages": cmp_messages,
            "turn": random.randint(0, 1),
        }

    elif state["cmp_hiding_place"] == "" or state["human_hiding_place"] == "":
        return {**state, "turn": (state["turn"] + 1) % 2}
    elif len(state["judge_messages"]) == 3:
        # both pennies have been hidden
        both_hidden_msg_content = "Both players have hidden their pennies"
        judge_messages.append(SystemMessage(content=both_hidden_msg_content))
        cmp_messages.append(SystemMessage(content=both_hidden_msg_content))
        turn = (state["turn"] + 1) % 2
        judge_messages.append(
            SystemMessage(
                content=judge_both_hidden_prompt.format(
                    current_player="computer" if turn == 0 else "human"
                )
            )
        )
        cmp_messages.append(
            SystemMessage(
                content=cmp_both_hidden_prompt.format(
                    current_player="your" if turn == 0 else "human's"
                )
            )
        )
        return {
            **state,
            "judge_messages": judge_messages,
            "cmp_messages": cmp_messages,
            "turn": turn,
        }
    else:
        # game has already begun ;)
        # someone just performed an action
        turn = state["turn"]

        response_stream = llm.stream(curr_messages + judge_messages)
        response = print_llm_stream(response_stream, print_reasoning=False)

        win = False
        if get_bool_tag(response, "PLAYER_WINS"):
            win = True
            response = "\n".join(response.split("\n")[:-2])

        judge_messages.append(AIMessage(content=response))
        cmp_messages.append(SystemMessage(content=response))

        if win:
            game_summary_prompt = SystemMessage(content=game_summary_prompt_text)
            response_stream = llm.stream(
                curr_messages + judge_messages + [game_summary_prompt]
            )
            response = print_llm_stream(response_stream, print_reasoning=False)
            judge_messages.append(AIMessage(content=response))
            cmp_messages.append(SystemMessage(content=response))

        return {
            **state,
            "judge_messages": judge_messages,
            "cmp_messages": cmp_messages,
            "turn": -1 if win else (turn + 1) % 2,
        }
