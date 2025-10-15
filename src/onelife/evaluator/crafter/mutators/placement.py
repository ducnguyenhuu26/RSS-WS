from .interface import Mutator
from ....typing_utils import implements
from crafter_oo.state_export import WorldState
from crafter_oo.constants import ActionT, CollectableT, collect
import random
from typing import cast
from crafter_oo.functional_env import transition
from onelife.evaluator.crafter.utils import MAP_ACTION_TO_INDEX
from loguru import logger
from typing_extensions import assert_never
from crafter_oo.constants import MaterialT, materials
from crafter_oo.state_export import Position
from crafter_oo.functional_env import reconstruct_world_from_state, export_world_state
from crafter_oo.testing_helpers import world_utils
from crafter_oo import objects as crafter_objects


PLACEMENT_ACTIONS: set[ActionT] = {
    "place_stone",
    "place_table",
    "place_furnace",
    "place_plant",
}


class PlaceIllegalItemMutator:
    def __init__(self):
        self.category = "Placement"
        self.logger = logger.bind(mutator=self.__class__.__name__)

    def precondition(self, state: WorldState, action: ActionT) -> bool:
        return action in PLACEMENT_ACTIONS

    def __call__(self, state: WorldState, action: ActionT) -> WorldState:
        mutated_state = state.model_copy(deep=True)

        # To create an illegal state, we will place a _different_ item than the one
        # that the player is trying to place.

        # Get the set of placement actions that were _not_
        # the placement action that was attempted
        other_placement_actions = PLACEMENT_ACTIONS - cast(set[ActionT], {action})

        # Choose a random other placement action
        other_placement_action = random.choice(list(other_placement_actions))

        self.logger.debug(
            f"Placement action {action} mutated to {other_placement_action}"
        )

        # Get the position of the targeted tile
        targeted_position = state.player.position + state.player.facing

        # We will mutate the targeted tile to the placement implied by
        # the mutated placement action.
        match other_placement_action:
            case "place_stone":
                mutated_state.set_tile_material(targeted_position, "stone")
            case "place_table":
                mutated_state.set_tile_material(targeted_position, "table")
            case "place_furnace":
                mutated_state.set_tile_material(targeted_position, "furnace")
            case "place_plant":
                # This actually involves adding a plant entity to the world
                world = reconstruct_world_from_state(mutated_state)
                # If there is an object on the tile we will delete it
                for obj in world._objects:
                    if obj is None:
                        continue
                    if (
                        obj.pos[0] == targeted_position.x
                        and obj.pos[1] == targeted_position.y
                    ):
                        self.logger.debug(
                            f"Removing object {obj} at {targeted_position} to make way for a plant"
                        )
                        world_utils.remove_object_from_world(world, obj)
                world_utils.add_object_to_world(
                    world,
                    crafter_objects.Plant,
                    (targeted_position.x, targeted_position.y),
                )
                mutated_state = export_world_state(
                    world, view=state.view, step_count=state.step_count
                )
        return mutated_state
