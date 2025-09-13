from typing import TypedDict
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    judge_messages: list[BaseMessage]
    cmp_messages: list[BaseMessage]
    human_hiding_place: str
    cmp_hiding_place: str
    scene_desc: str
    human_powerups: str
    cmp_powerups: str
    turn: int


initialState = AgentState(
    judge_messages=[],
    cmp_messages=[],
    human_hiding_place="",
    cmp_hiding_place="",
    scene_desc="",
    human_powerups="",
    cmp_powerups="",
    turn=-1,
)
