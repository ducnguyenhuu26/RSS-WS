from crafter.state_export import WorldState
import jsonpatch
from typing import Protocol
from crafter.functional_env import (
    reconstruct_world_from_state,
    export_world_state,
)
from crafter.state_export import WorldState
from crafter.constants import ActionT
from .utils import find_player
from crafter.testing_helpers import (
    player_utils,
    world_utils,
)
from crafter.functional_env import initial_state
from crafter import objects
from ...typing_utils import implements
from crafter.env import Env
from crafter.functional_env import (
    EnvConfig,
    transition,
    initial_state,
    export_world_state,
)
from .utils import get_world_state
import random
from ..core import SymbolicTransition
from .utils import MAP_ACTION_TO_INDEX


# Note: This is almost a copy of the format_state function used to generate
# training data for the neural world model in e0008, except we _do not_ exclude
# the materials field.
def _gamestate_to_json(state: WorldState) -> dict:
    excluded_fields = {"event_bus", "serialized_random_state"}

    serialized_state = state.model_dump(exclude=excluded_fields)

    def format_serialized_state(serialized_state: dict) -> dict:
        # Remove the player field from the .objects list, so it isn't duplicated
        # since it is already in the .player field.
        serialized_state["objects"] = [
            obj for obj in serialized_state["objects"] if obj["name"] != "player"
        ]

        # Sort the objects by entity_id
        serialized_state["objects"] = sorted(
            serialized_state["objects"], key=lambda x: x["entity_id"]
        )

        # Sort the chunks by chunk_key
        serialized_state["chunks"] = sorted(
            serialized_state["chunks"], key=lambda x: x["chunk_key"]
        )

        # For each chunk, sort the objects within the chunk
        for chunk in serialized_state["chunks"]:
            chunk["objects"] = sorted(chunk["objects"])

        return serialized_state

    return format_serialized_state(serialized_state)


class JSONPatchEditDistance:
    def __call__(self, state1: WorldState, state2: WorldState) -> int:
        json1 = _gamestate_to_json(state1)
        json2 = _gamestate_to_json(state2)
        patch = jsonpatch.make_patch(json1, json2)
        return len(list(patch))


class RandomMovementPolicy:
    def __init__(self, policy_seed: int, num_transitions: int):
        self.policy_rng = random.Random(policy_seed)
        self.movement_actions: list[ActionT] = [
            "move_left",
            "move_right",
            "move_up",
            "move_down",
        ]
        self.num_transitions = num_transitions

    def __call__(self) -> list[SymbolicTransition[WorldState]]:

        transitions: list[SymbolicTransition[WorldState]] = []

        config = EnvConfig(
            view=(9, 9),
            size=(64, 64),
            seed=1,
        )

        state = initial_state(area=config.size, view=config.view, seed=config.seed)

        for _ in range(self.num_transitions):
            action = self.policy_rng.choice(self.movement_actions)
            prev_state = state

            state, _ = transition(state, MAP_ACTION_TO_INDEX[action])

            transitions.append(SymbolicTransition(prev_state, action, state))

        return transitions
