from crafter.state_export import WorldState
from crafter.functional_env import (
    initial_state,
    reconstruct_world_from_state,
    export_world_state,
    transition,
)
import copy
from crafter.testing_helpers import player_utils, world_utils
from onelife.evaluator.crafter.utils import MAP_ACTION_TO_INDEX
import pytest
from crafter.constants import ActionT
from crafter import engine as crafter_engine

from onelife.evaluator.crafter.utils import find_player
import numpy as np
from crafter.constants import actions
from crafter.constants import MaterialT
from crafter import objects as crafter_objects
from crafter.constants import actions
from typing import cast
from onelife.evaluator.crafter.mutators.placement import (
    PLACEMENT_ACTIONS,
    PlaceIllegalItemMutator,
)
from crafter.state_export import PlantState
from loguru import logger


def make_placement_state(
    add_entity_on_target_tile: bool = False,
) -> tuple[WorldState, crafter_engine.World]:
    view = (9, 9)
    state = initial_state(area=(9, 9), view=view, seed=1)
    world = reconstruct_world_from_state(state)

    player = find_player(world)
    player_utils.set_player_position(player, (5, 5))

    # Clear all the other tiles around the world to be grass
    for x in range(view[0]):
        for y in range(view[1]):
            world_utils.set_tile_material(world, (x, y), "grass")

    if add_entity_on_target_tile:
        world_utils.add_object_to_world(world, crafter_objects.Cow, (6, 5))

    # Make the player face the target material
    player_utils.set_player_facing(player, (1, 0))

    # Give the player required resources to place anything
    player_utils.set_player_inventory_item(player, "stone", 6)
    player_utils.set_player_inventory_item(player, "sapling", 6)
    player_utils.set_player_inventory_item(player, "wood", 6)
    player_utils.set_player_inventory_item(player, "iron", 6)

    return export_world_state(world, view=view, step_count=0), world


@pytest.mark.parametrize(
    "action",
    list(PLACEMENT_ACTIONS),
)
def test_mutator(action: ActionT):
    collection_state, _ = make_placement_state()

    mutator = PlaceIllegalItemMutator()

    assert mutator.precondition(collection_state, action)

    true_next_state, _ = transition(
        copy.deepcopy(collection_state), MAP_ACTION_TO_INDEX[action]
    )

    mutated_next_state = mutator(collection_state, action)

    # Check to see if the mutator placed a plant in the world state.
    # If so, the materials check can be skipped because the material
    # will not have changed.
    _, target_entity = mutated_next_state.get_target_tile()
    match target_entity:
        case PlantState():
            logger.info(
                f"Plant placed at {target_entity.position}, marking test as successful"
            )
            # The test is a success in this case.
            pass  # We can skip the materials check because a plant has been added
        case _:
            assert mutated_next_state.materials != true_next_state.materials

    # We do this check just to confirm that the __eq__ method is working
    # correctly.
    assert true_next_state.materials == copy.deepcopy(true_next_state.materials), (
        "The __eq__ method on materials should assert equality when the materials are the same"
        " but this is not the case; this test is invalid."
    )


@pytest.mark.parametrize(
    "action",
    list(PLACEMENT_ACTIONS),
)
def test_mutator_with_entity(action: ActionT):
    collection_state, _ = make_placement_state(add_entity_on_target_tile=True)

    mutator = PlaceIllegalItemMutator()

    assert mutator.precondition(collection_state, action)

    true_next_state, _ = transition(
        copy.deepcopy(collection_state), MAP_ACTION_TO_INDEX[action]
    )

    mutated_next_state = mutator(collection_state, action)

    # Check to see if the mutator placed a plant in the world state.
    # If so, the materials check can be skipped because the material
    # will not have changed.
    _, target_entity = mutated_next_state.get_target_tile()
    match target_entity:
        case PlantState():
            logger.info(
                f"Plant placed at {target_entity.position}, marking test as successful"
            )
            # The test is a success in this case.
            pass  # We can skip the materials check because a plant has been added
        case _:
            assert mutated_next_state.materials != true_next_state.materials

    # We do this check just to confirm that the __eq__ method is working
    # correctly.
    assert true_next_state.materials == copy.deepcopy(true_next_state.materials), (
        "The __eq__ method on materials should assert equality when the materials are the same"
        " but this is not the case; this test is invalid."
    )


NON_PLACEMENT_ACTIONS = set(actions) - PLACEMENT_ACTIONS


@pytest.mark.parametrize(
    "action",
    list(NON_PLACEMENT_ACTIONS),
)
def test_mutator_inactive_on_non_placement_actions(action: ActionT):
    collection_state, _ = make_placement_state()

    mutator = PlaceIllegalItemMutator()

    assert not mutator.precondition(collection_state, action)
