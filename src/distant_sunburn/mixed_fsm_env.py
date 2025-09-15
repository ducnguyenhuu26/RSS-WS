from pydantic import BaseModel
from enum import StrEnum
from copy import deepcopy
from dataclasses import dataclass
import random


class Action(StrEnum):
    TOGGLE_DETERMINISTIC_SWITCH = "toggle_deterministic_switch"
    TOGGLE_STOCHASTIC_SWITCH = "toggle_stochastic_switch"
    TOGGLE_STATIC_SWITCH = "toggle_static_switch"


@dataclass
class State:
    deterministic_switch: int
    stochastic_switch: int
    static_switch: int
    rng_state: random.Random


def initial_state(seed: int) -> State:
    return State(
        deterministic_switch=0,
        stochastic_switch=0,
        static_switch=0,
        rng_state=random.Random(seed),
    )


def transition_function(state: State, action: Action) -> State:
    state = deepcopy(state)
    if action == Action.TOGGLE_DETERMINISTIC_SWITCH:
        state.deterministic_switch = int(not state.deterministic_switch)
    elif action == Action.TOGGLE_STOCHASTIC_SWITCH:
        state.stochastic_switch = int(state.rng_state.random() < 0.5)
    elif action == Action.TOGGLE_STATIC_SWITCH:
        pass  # Static switch does nothing
    return state
