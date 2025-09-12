from langgraph.graph import StateGraph, END
from typing import Iterator, TypedDict, Sequence, Annotated
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ChatMessage, AIMessage, BaseMessageChunk
from langchain_ollama import ChatOllama
import random

class AgentState(TypedDict):
    judge_messages: list[BaseMessage]
    cmp_messages: list[BaseMessage]
    human_hiding_place: str
    cmp_hiding_place: str
    scene_desc: str
    human_powerups: str
    cmp_powerups: str
    turn: int

llm = ChatOllama(model='qwen3:8b', reasoning=True)


# def print_state(state: AgentState):
#     print("Current State:")
#     print("Judge Messages: ", [i.content for i in state["judge_messages"]])
#     print("Computer Messages: ", [i.content for i in state["cmp_messages"]])
#     print("Turn: ", state["turn"])


game_rules = """
'Hide the Penny' game rules:
1. There are two players ('human' and 'computer') and a 'judge'.
2. The 'human' and 'computer' play the game, while the 'judge' just mediates and drives the game.
3. Initially a scene is set and the two players hide their respective pennies within that scene.
4. The goal of the game is to find the other player's penny before they find yours.
5. The players take turns performing 'actions'.
6. An 'action' can be any action the current player could do which is possible within the scene and based on the previous history.
7. The judge decides what happens after each action and what the outcome is.
8. The player who finds the other player's penny first wins the game.
9. Finding your own penny is not worth anything, obviously.
"""

# game_rules_str = """
# 'Hide the Penny' game rules:
# 1. There are two players and one judge. One of the players is referred to as "computer" and one is referred to as "human". The judge doesn't participate in the game, just mediates and conducts it.
# 2. The judge sets a scene, for eg - 'A coffee shop interior with a few tables, a barista counter and a cash counter'. This description can also be augmented by arbitrary details and can be enriched as the game progresses.
# 3. The two players each are given a penny that they have to hide somewhere in this scene. They can invent some details within the realm of possibility for hiding their pennies and inform the judge of this location.
# 4. The players race to find the other player's penny. There is no point in finding your own penny!
# 4. Then the players take turns performing actions. Each action can be as simple as just looking under a table, or very complex, such as asking a fox to help, for eg. It is entirely up to the judge to the determine the outcome of each action.
# 5. The judge and the players can't cheat. They can't change the location of their pennies and try to mislead the other player. If one player looks in a place, or performs an action that logically reveals the location of the other players' penny, then the judge has to give them the win.
# 6. The player who finds the other player's penny first wins.
# """

judge_system_prompt = (
    "You and two other players are playing 'Hide the Penny'. "
    f"{game_rules} "
    "You are the 'Judge' in this scenario"
    "Your task is as follows: "
    # "1. Set the scene where the game will take place. "
    "1. Remember the locations of the pennies of the 'human' and the 'computer' players "
    "2. The players perform actions one-by-one. You need to decide what happens. "
    "3. NEVER REVEAL THE LOCATIONS OF THE PENNIES IN ANY CASE. "
    "4. If the current player performs an action that will logically lead to them finding the other player's penny, then they win the game. Signal that by ending your message with <PLAYER_WINS/> "
    "5. If the current player doesn't find the penny, the game just continues, and DON'T PUT <PLAYER_WINS /> IN YOUR RESPONSE."
    "6. If the current player's action doesn't logically lead to them finding the other penny, but instead has a chance of that, "
    "weigh the options and decide accordingly, whether the player wins or not. Signal the win using <PLAYER_WINS />  "
    "7. If the current player's action is irrelevant or doesn't make sense, just say that nothing happens. "
    "8. You can keep your responses a bit descriptive and engaging, but don't make them too long. "
    "9. Don't repeat or paraphrase the action performed by the current player. Just decide what happens next based on that action. Inform the players of any important details that get revealed due to the action, but don't reference the hidden locations of the pennies. "
    "10. No need to check if the current player finds their own penny, that doesn't count. "
    "11. Always respond in the format 'JUDGE: <your response here>' "
    "12. Only give the judge's response, don't include anything else. "
    "DON'T ASSUME ANY OTHER ROLE THAN THIS "
)

cmp_system_prompt = (
    "You and one other player are playing 'Hide the Penny'. "
    f"{game_rules} "
    "You are 'computer', and your opponent is 'human'. "
    "The 'judge' is another person (not a player) who's conducting the game. "
    "Your task is as follows: "
    # "1. First you'll be asked to hide your penny somewhere in the scene set by the judge. "
    # "2. You can be as vague or as specific as you want. Feel free to invent details, as long as they lie within the realm of possibility for that scene, and clearly call out any details you do invent. "
    "1. You and 'human' will take turns to try to find the other's penny "
    "2. DON'T REVEAL YOUR PENNY'S LOCATION TO THE HUMAN"
    "3. You perform actions one-by-one and the judge decides what happens. "
    "4. If you find the human's penny before he finds your penny, you win! "
    "5. If the 'human' finds your penny before you find his, he wins. "
    "6. Keep in mind that there is no point in finding your own penny, that doesn't count. "
    "7. Always respond in the format 'COMPUTER: <your response here>' "
    "8. You can keep your responses a bit descriptive and engaging, but all of it has to be just one action and the outcome is decided by the judge, not you."
    "9. DON'T TRY TO DETERMINE THE OUTCOME OF YOUR ACTION, THE JUDGE WILL DO THAT. JUST STATE YOUR ACTION."
    "10. Only give the computer's response, don't include anything else. "
    "DON'T ASSUME ANY OTHER ROLE THAN THIS "
)

def print_llm_stream(stream: Iterator[BaseMessageChunk], print_reasoning = True, print_response = True):
    response = ""
    if print_reasoning:
        print("THINKING START\n")
    for i in stream:
        if "reasoning_content" in i.additional_kwargs.keys() and print_reasoning:
            print(i.additional_kwargs['reasoning_content'], end="")
        else:
            if len(response) == 0 and print_reasoning:
                print("\n\nTHINKING END")
            response += str(i.content)
            if print_response:  
                print(i.content, end="")
    print("")
    return response

# def print_stream(stream):
#     prev_len = 0
#     for s in stream:
#         print("human hid here: ", s["human_hiding_place"])
#         print("computer hid here: ", s["cmp_hiding_place"])
#         if len(s["messages"]) == 0 or len(s["messages"]) == prev_len: 
#             continue
#         message = s["messages"][-1]
#         prev_len = len(s["messages"])
#         if isinstance(message, tuple):
#             print(message)
#         elif message.role != "system":
#             print(f"\n========================\n{message.role.upper()}: {message.content}\n========================\n")

def get_tag(s: str, tag: str):
    find_str_open = "<" + tag + ">"
    find_str_close = "</" + tag + ">"
    open_idx = s.find(find_str_open) + len(tag) + 2
    close_idx = s.find(find_str_close)
    return s[open_idx: close_idx]

def get_bool_tag(s: str, tag:str):
    find_str = "<" + tag + "/>"
    idx = s.find(find_str)
    return idx != -1


def judge(state: AgentState) -> AgentState:
    print("==========JUDGE TURN============")
    # print_state(state)
    # system_prompt = ChatMessage(role="system", content=(
    #     "You are the judge in a game of Hide the Penny" \
    #     "There are two players and they will hide their pennies and then try to find the other's penny" \
    #     "You are going to conduct this game and decide what happens with the players' actions"
    # ))
    system_prompt = ChatMessage(role="system", content=judge_system_prompt)
    curr_messages = [system_prompt]

    judge_messages = state["judge_messages"]
    cmp_messages = state["cmp_messages"]

    if len(state['judge_messages']) == 0:
        # game hasn't started yet
        # set it up
        scene_setting_prompt = ChatMessage(role="system", content=(
        "It is the start of the game. You need to decide what will be the scene in which the game will take place." \
        "You can be as descriptive, or arbitrary as you want. Just set the scene for the players to work with." \
        "The players will hide their pennies somewhere in your scene, and then try to find the other's penny." \
        "Return the scene description enclosed in <SCENE> and </SCENE>." \
        "Remember: Don't include anything other than the scene description in these tags!"
        ))
        response_stream = llm.stream(curr_messages + [scene_setting_prompt])
        response = print_llm_stream(response_stream, print_reasoning=False)

        scene_desc = get_tag(str(response), "SCENE")
        scene_str = f"JUDGE: The scene is set as follows: \n{scene_desc}\n"

        judge_messages.append(AIMessage(content=scene_str))
        cmp_messages.append(SystemMessage(content=scene_str))


        # print(scene_msg.content)
        return {
            **state,
            "scene_desc": scene_desc,
            "judge_messages": judge_messages,
            "cmp_messages": cmp_messages,
            "turn": random.randint(0,1)
        }
    
    elif state["cmp_hiding_place"] == "" or state["human_hiding_place"] == "":
        return {
            **state,
            "turn": (state["turn"] + 1) % 2
        }
    elif len(state["judge_messages"]) == 3:
        # both pennies have been hidden
        both_hidden_msg_content = "Both players have hidden their pennies"
        judge_messages.append(SystemMessage(content=both_hidden_msg_content))
        cmp_messages.append(SystemMessage(content=both_hidden_msg_content))
        turn = (state["turn"] + 1)%2
        judge_messages.append(SystemMessage(content=f"Now the players take turns to find the other's penny. It is currently {'computer' if turn == 0 else 'human'}'s turn."))
        cmp_messages.append(SystemMessage(content=f"Now you and human take turns to find the other's penny. It is currently {'your' if turn == 0 else 'the human\'s'} turn."))
        return {
            **state,
            "judge_messages": judge_messages,
            "cmp_messages": cmp_messages,
            "turn": turn
        }
    else:
        # game has already begun ;)
        # someone just performed an action
        turn = state["turn"]
        
        # turn_prompt = ChatMessage(role="system", content=f"It is currently {'computer' if turn == 0 else 'human'}'s turn.")
        # hidden_location_prompt = ChatMessage(role="system", content=f"REMEMBER: human's penny is hidden here: {state["human_hiding_place"]}\n and computer's penny is hidden here: {state["cmp_hiding_place"]}\n")
    
        response_stream = llm.stream(curr_messages + judge_messages)
        response = print_llm_stream(response_stream, print_reasoning=False)
        
        # print("JUDGE_PROMPT: ", curr_messages + state["messages"])
        win = False
        if get_bool_tag(response, "PLAYER_WINS"):
            win = True
            response = "\n".join(response.split("\n")[:-2])
        
        judge_messages.append(AIMessage(content=response))
        cmp_messages.append(SystemMessage(content=response))

        if win: 
            game_summary_prompt = SystemMessage(content=(
                "The game is over. Please provide a summary of the game, from the initial scene setting to the end, "
                "describing the key moments where the players were close to finding the other pennies. "
                "DON'T REPEAT YOUR LAST RESPONSE. "
                "Highlight the actions that led to the conclusion of the game. "
                "Keep it as descriptive and engaging as possible. "
                "Tell it in a storytelling manner. "
                "This summary will be read out to both players. "
                "It is the last message you will send, so at the end, say goodbye to the players. "
                "It's your last message, so make it count! "
            ))
            response_stream = llm.stream(curr_messages + judge_messages + [game_summary_prompt])
            response = print_llm_stream(response_stream, print_reasoning=False)
            judge_messages.append(AIMessage(content=response))
            cmp_messages.append(SystemMessage(content=response))

        return {
            **state,
            "judge_messages": judge_messages,
            "cmp_messages": cmp_messages,
            "turn": -1 if win else (turn + 1)%2
        }


def cmp_agent(state: AgentState) -> AgentState:
    print("==========COMPUTER TURN============")
    # print_state(state)

    system_prompt = ChatMessage(role="system", content=cmp_system_prompt)
    curr_messages = [system_prompt]

    judge_messages = state["judge_messages"]
    cmp_messages = state["cmp_messages"]

    if state["cmp_hiding_place"] == "":
        # hasn't hidden the penny yet
        hiding_prompt = ChatMessage(role="system", content=("You need to hide a penny in the scene described by the user." 
                                                            "Return a short description of the location and way you would hide the penny," 
                                                            "enclose it in <CMP_HIDING_PLACE> and </CMP_HIDING_PLACE>"))
        
        response_stream = llm.stream(curr_messages + cmp_messages + [hiding_prompt])
        response = print_llm_stream(response_stream, print_reasoning=False, print_response=False)

        cmp_hiding_place = get_tag(str(response), "CMP_HIDING_PLACE")
        response_hiding_place = f"COMPUTER: I have hidden my penny here: {cmp_hiding_place}\n"

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
        # setup_prompt = ChatMessage(role="system", content=(
        #     "You have already hidden the penny"
        #     "Now you must try to find human's penny"
        # ))
        # already_hidden_prompt = ChatMessage(role="system", content=f"You have already hidden your penny as follows: {state["cmp_hiding_place"]} . Now you have got to find human's penny before he finds yours.")
        # action_prompt = SystemMessage(content=(
        #     "JUDGE: It is your turn computer, what will your next action be? "
        # ))
        action_response_stream = llm.stream(curr_messages + cmp_messages)
        action_response = print_llm_stream(action_response_stream, print_reasoning=False)

        judge_messages.append(HumanMessage(content=action_response))
        cmp_messages.append(AIMessage(content=action_response))

        return {
            **state,
            "judge_messages": judge_messages,
            "cmp_messages": cmp_messages,
        }

def human_agent(state: AgentState) -> AgentState:
    print("==========YOUR TURN============")
    judge_messages = state["judge_messages"]
    cmp_messages = state["cmp_messages"]

    if state["human_hiding_place"] == "":
        hiding_place = input("Where do you want to hide your penny? ")
        judge_messages.append(HumanMessage(content=f"HUMAN: I have hidden my penny here: {hiding_place}\n"))

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
    
    

def turn_router(state:AgentState) -> str:
    if state["turn"] == 0:
        return "cmp"
    elif state["turn"] == 1:
        return "human"
    else:
        return "end"
    

graph = StateGraph(AgentState)
graph.add_node('judge', judge)
graph.add_node('human_agent', human_agent)
graph.add_node('cmp_agent', cmp_agent)
graph.set_entry_point('judge')
graph.add_conditional_edges('judge', turn_router, {
    "cmp": "cmp_agent",
    "human": "human_agent",
    "end": END
})
graph.add_edge("cmp_agent", "judge")
graph.add_edge("human_agent", "judge")

app = graph.compile()
print(app.get_graph().draw_mermaid())

app.invoke({
    "judge_messages": [],
    "cmp_messages": [],
    "human_hiding_place": "",
    "cmp_hiding_place": "",
    "cmp_powerups": "",
    "human_powerups": "",
    "scene_desc": "",
    "turn": -1
})






        
    