from typing import TypedDict


class AgentState(TypedDict):
    history: str
    game_started: bool
    human_hiding_place: str
    cmp_hiding_place: str
    scene_desc: str
    turn: int


initialState = AgentState(
    history="",
    human_hiding_place="",
    cmp_hiding_place="",
    scene_desc="",
    turn=-1,
    game_started=False,
)
