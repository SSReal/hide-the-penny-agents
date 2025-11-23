from src.state import AgentState
from src.llm import llm
from src.prompts.game import system_prompt


def cmp_agent(state: AgentState) -> AgentState:
    print("==========COMPUTER TURN============")

    if state["cmp_hiding_place"] == "":
        prompt = (
            system_prompt
            + state["history"]
            + (
                "\n\nThe Judge has set the scene as described above."
                "\nThe computer will now tell the judge where they're hiding their penny."
                "\n\nCOMPUTER: "
            )
        )

        response_stream = llm.generate_computer_stream(prompt)
        full_res = ""
        for chunk in response_stream:
            # print(chunk, end="") # don't print it, obviously
            full_res += chunk

        label = full_res.find("COMPUTER: ")
        label = label if label > -1 else -10
        state["cmp_hiding_place"] = full_res[label + 10 :]
        # don't add this to history, cuz, well, obviously
        state["history"] += "\n\n COMPUTER: I have hidden my penny!\n\n"

    else:
        # The game has already begun ;)
        response_stream = llm.generate_computer_stream(
            system_prompt
            + state["history"]
            + "\n\n It's now computer's turn. The computer thinks hard on how to find the human's penny."
            "\n\nCOMPUTER: "
        )

        full_res = ""
        for chunk in response_stream:
            print(chunk, end="")
            full_res += chunk

        state["history"] += full_res + "\n\n"
    print("")
    return state
