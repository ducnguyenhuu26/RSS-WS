from pydantic import BaseModel
from enum import StrEnum
from copy import deepcopy


class Action(StrEnum):
    TOGGLE_A = "toggle_a"
    TOGGLE_B = "toggle_b"


class State(BaseModel):
    switch_a: int
    switch_b: int


def transition_function(state: State, action: Action) -> State:
    state = deepcopy(state)
    if action == Action.TOGGLE_A:
        state.switch_a = not state.switch_a
    elif action == Action.TOGGLE_B:
        state.switch_b = not state.switch_b
    return state


def initial_state() -> State:
    return State(switch_a=0, switch_b=1)
