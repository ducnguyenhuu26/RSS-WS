from ....typing_utils import implements
from .v1 import Mutator
from crafter.state_export import WorldState, Position
from crafter.constants import ActionT
import random


class IllegalMovementMutator:
    def __init__(self):
        self.category = "Physics"

    def precondition(self, state: WorldState, action: ActionT) -> bool:
        """
        Produce a movement on an action that ordinarily does not result in a movement.
        """

        non_movement_actions: set[ActionT] = {
            "noop",
            "do",
            "sleep",
            "place_stone",
            "place_table",
            "place_furnace",
            "place_plant",
            "make_wood_pickaxe",
            "make_stone_pickaxe",
            "make_iron_pickaxe",
            "make_wood_sword",
            "make_stone_sword",
            "make_iron_sword",
        }

        return action in non_movement_actions

    def __call__(self, state: WorldState, action: ActionT) -> WorldState:
        directions = [
            Position(x=0, y=1),
            Position(x=1, y=0),
            Position(x=0, y=-1),
            Position(x=-1, y=0),
        ]
        mutated_state = state.model_copy(deep=True)
        mutated_state.player.facing = random.choice(directions)

        mutated_state.player.position += mutated_state.player.facing

        return mutated_state
