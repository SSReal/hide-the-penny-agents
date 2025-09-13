from src.prompts.game import game_rules
from langchain_core.prompts import ChatMessagePromptTemplate

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

setup_prompt = (
    "It is the start of the game. You need to decide what will be the scene in which the game will take place."
    "You can be as descriptive, or arbitrary as you want. Just set the scene for the players to work with."
    "The players will hide their pennies somewhere in your scene, and then try to find the other's penny."
    "Return the scene description enclosed in <SCENE> and </SCENE>."
    "Remember: Don't include anything other than the scene description in these tags!"
)

both_hidden_prompt = "Both players have hidden their pennies. Now the players take turns to find the other's penny. It is currently {current_player}'s turn."

game_summary_prompt_text = (
    "The game is over. Please provide a summary of the game, from the initial scene setting to the end, "
    "describing the key moments where the players were close to finding the other pennies. "
    "DON'T REPEAT YOUR LAST RESPONSE. "
    "Highlight the actions that led to the conclusion of the game. "
    "Keep it as descriptive and engaging as possible. "
    "Tell it in a storytelling manner. "
    "This summary will be read out to both players. "
    "At the end, say goodbye to the players. "
    "It's your last message, so make it count! "
)
