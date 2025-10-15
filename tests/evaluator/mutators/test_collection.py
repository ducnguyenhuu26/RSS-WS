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
from onelife.evaluator.crafter.mutators.collection import (
    CollectIllegalMaterialMutator,
    TILE_TO_MATERIALS,
)
from onelife.evaluator.crafter.utils import find_player
import numpy as np
from crafter.constants import actions
from crafter.constants import MaterialT
from crafter import objects as crafter_objects
from crafter.constants import actions
from typing import cast


def make_collection_state(
    target_material: MaterialT,
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

    # Set the tile to the right of the player to the target material for collection
    world_utils.set_tile_material(world, (6, 5), target_material)

    if add_entity_on_target_tile:
        world_utils.add_object_to_world(world, crafter_objects.Cow, (6, 5))

    # Make the player face the target material
    player_utils.set_player_facing(player, (1, 0))

    # Give the player required resources to collect anything
    player_utils.set_player_inventory_item(player, "iron_pickaxe", 1)

    return export_world_state(world, view=view, step_count=0), world


@pytest.mark.parametrize(
    "material",
    list(TILE_TO_MATERIALS.keys()),
)
def test_collection_illegal_material_mutator(material: MaterialT):
    collection_state, _ = make_collection_state(material)

    mutator = CollectIllegalMaterialMutator()

    assert mutator.precondition(collection_state, "do")

    true_next_state, _ = transition(
        copy.deepcopy(collection_state), MAP_ACTION_TO_INDEX["do"]
    )

    mutated_next_state = mutator(collection_state, "do")

    # Now we compare the two inventories. The mutated inventory
    # should have an item that the true next state does not, and
    # additionally, the mutation will not have consumed any resources.

    assert mutated_next_state.player.inventory != true_next_state.player.inventory

    # We do this check just to confirm that the __eq__ method is working
    # correctly.
    assert true_next_state.player.inventory == copy.deepcopy(
        true_next_state.player.inventory
    ), (
        "The __eq__ method on inventories should assert equality when the inventories are the same"
        " but this is not the case; this test is invalid."
    )


# We could do this in one pytest.mark.parametrize, but it's clearer to
# separate these so the parameterization is easier to reads.
@pytest.mark.parametrize("material", list(TILE_TO_MATERIALS.keys()))
def test_collection_illegal_material_mutator_with_entity(material: MaterialT):
    collection_state, _ = make_collection_state(
        material, add_entity_on_target_tile=True
    )

    mutator = CollectIllegalMaterialMutator()

    assert mutator.precondition(collection_state, "do")

    true_next_state, _ = transition(
        copy.deepcopy(collection_state), MAP_ACTION_TO_INDEX["do"]
    )

    mutated_next_state = mutator(collection_state, "do")

    # Now we compare the two inventories. The mutated inventory
    # should have an item that the true next state does not, and
    # additionally, the mutation will not have consumed any resources.

    assert mutated_next_state.player.inventory != true_next_state.player.inventory

    # We do this check just to confirm that the __eq__ method is working
    # correctly.
    assert true_next_state.player.inventory == copy.deepcopy(
        true_next_state.player.inventory
    ), (
        "The __eq__ method on inventories should assert equality when the inventories are the same"
        " but this is not the case; this test is invalid."
    )


@pytest.mark.parametrize("action", list(set(actions) - cast(set[ActionT], {"do"})))
def test_collection_illegal_material_mutator_inactive_on_non_do_actions(
    action: ActionT,
):
    collection_state, _ = make_collection_state("grass", add_entity_on_target_tile=True)

    mutator = CollectIllegalMaterialMutator()

    assert not mutator.precondition(collection_state, action)
