from src.state import AgentState
import random
from src.llm import llm
from src.prompts.game import system_prompt


def judge(state: AgentState) -> AgentState:
    print("==========JUDGE TURN============")

    if state["scene_desc"] == "":
        # scene hasn't been set yet
        scene_setting_prompt = system_prompt + (
            "The judge provides the description of the scene the players have to work with, "
            "and ends by saying HIDE YOUR PENNIES! (uppercase and word-by-word)."
            "\n\nJUDGE: "
        )
        response_stream = llm.generate_judge_stream(scene_setting_prompt)
        full_res = ""
        for chunk in response_stream:
            full_res += chunk
            print(chunk, end="")

        msg_end = full_res.find("HIDE YOUR PENNIES!")
        if msg_end > -1:
            state["scene_desc"] = full_res[:msg_end]
        else:
            print("fallback")
            # fallback
            state["scene_desc"] = full_res

        state["history"] += full_res + "\n\n"
        state["turn"] = random.randint(0, 1)

    elif state["human_hiding_place"] == "" or state["cmp_hiding_place"] == "":
        state["turn"] = (state["turn"] + 1) % 2

    elif not state["game_started"]:
        state["game_started"] = True
        state["history"] += "JUDGE: Let the game begin!!!\n\n"
        state["turn"] = (state["turn"] + 1) % 2
    else:
        # game has begun
        response_stream = llm.generate_judge_stream(
            system_prompt
            + state["history"]
            + f"\n\n (The judge knows that the {"human" if state["turn"] == 0 else "computer"} has hidden their penny here:"
            f" {state["human_hiding_place"] if state["turn"] == 0 else state["cmp_hiding_place"]})\n"
            f"Now the judge decides what's going to happen for {"computer" if state["turn"] == 0 else "human"}'s turn."
            "\n\n JUDGE: "
        )

        full_res = ""
        for chunk in response_stream:
            print(chunk, end="")
            full_res += chunk

        state["history"] += full_res + "\n\n"
        msg_end = full_res.find("<end>")
        if msg_end > -1:
            state["turn"] = -1
        else:
            state["turn"] = (state["turn"] + 1) % 2
    print("")
    return state
