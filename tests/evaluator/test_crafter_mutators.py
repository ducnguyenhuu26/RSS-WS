"""
Tests for Crafter mutators.

These tests verify that mutators correctly apply their intended modifications
to WorldState objects without errors.
"""

import pytest
from crafter.state_export import WorldState, CowState, Position
from crafter.functional_env import (
    initial_state,
    reconstruct_world_from_state,
    export_world_state,
)
from crafter.testing_helpers import player_utils, world_utils
from crafter import objects

from distant_sunburn.evaluator.crafter.mutators import (
    AddIllegalItemMutator,
    TeleportEntityToIllegalTileMutator,
)
from distant_sunburn.evaluator.crafter.utils import (
    find_all_objects_for_type,
    find_object_in_state,
)


def create_test_state_with_cow_and_wall() -> WorldState:
    """Create a test state with a cow and a stone wall."""
    view = (9, 9)
    state = initial_state(area=(9, 9), view=view, seed=1)
    world = reconstruct_world_from_state(state)

    # Add a cow at position (3, 3)
    cow = objects.Cow(world, (3, 3))
    world.add(cow)

    # Set a stone wall at position (0, 0)
    world_utils.set_tile_material(world, (0, 0), "stone")

    return export_world_state(world, view=view, step_count=0)


def test_add_illegal_item_mutator():
    """Test that AddIllegalItemMutator correctly adds items to inventory."""
    # 1. Arrange: Create a state with a player
    view = (9, 9)
    state = initial_state(area=(9, 9), view=view, seed=1)
    world = reconstruct_world_from_state(state)
    initial_state_obj = export_world_state(world, view=view, step_count=0)

    # 2. Act
    mutator = AddIllegalItemMutator("wood", 2)

    # Ensure precondition passes
    assert mutator.precondition("noop", initial_state_obj)

    mutated_state = mutator(initial_state_obj)

    # 3. Assert
    # Check that wood was added
    assert mutated_state.player.inventory.wood == 2

    # Check that original state is unchanged
    assert initial_state_obj.player.inventory.wood == 0


def test_teleport_entity_mutator():
    """Test that TeleportEntityToIllegalTileMutator moves cows to illegal tiles."""
    # 1. Arrange: Create a state with a cow and a stone wall
    initial_state_obj = create_test_state_with_cow_and_wall()
    cow = find_all_objects_for_type(initial_state_obj, CowState)[0]
    original_cow_pos = cow.position

    # 2. Act
    mutator = TeleportEntityToIllegalTileMutator(seed=42)

    # Ensure precondition passes
    assert mutator.precondition("noop", initial_state_obj)

    mutated_state = mutator(initial_state_obj)

    # 3. Assert
    # Find the cow in the new state
    mutated_cow = find_object_in_state(mutated_state, cow.entity_id, CowState)
    assert mutated_cow is not None

    # Assert the cow has moved to an illegal tile
    assert mutated_cow.position != original_cow_pos
    material, _ = mutated_state.get_tile(mutated_cow.position)
    assert material == "stone"

    # Assert original state is unchanged
    original_cow_still_in_place = find_object_in_state(
        initial_state_obj, cow.entity_id, CowState
    )
    assert original_cow_still_in_place is not None
    assert original_cow_still_in_place.position == original_cow_pos


def test_teleport_entity_mutator_precondition_no_cow():
    """Test that TeleportEntityToIllegalTileMutator precondition fails when no cow exists."""
    # 1. Arrange: Create a state without any cows
    view = (9, 9)
    state = initial_state(area=(9, 9), view=view, seed=1)
    world = reconstruct_world_from_state(state)
    initial_state_obj = export_world_state(world, view=view, step_count=0)

    # 2. Act & Assert
    mutator = TeleportEntityToIllegalTileMutator(seed=42)
    assert not mutator.precondition("noop", initial_state_obj)


def test_teleport_entity_mutator_precondition_no_illegal_tiles():
    """Test that TeleportEntityToIllegalTileMutator precondition fails when no illegal tiles exist."""
    # 1. Arrange: Create a state with a cow but only walkable tiles
    view = (9, 9)
    state = initial_state(area=(9, 9), view=view, seed=1)
    world = reconstruct_world_from_state(state)

    # Add a cow
    cow = objects.Cow(world, (3, 3))
    world.add(cow)

    # Ensure all tiles are walkable (grass)
    for x in range(9):
        for y in range(9):
            world_utils.set_tile_material(world, (x, y), "grass")

    initial_state_obj = export_world_state(world, view=view, step_count=0)

    # 2. Act & Assert
    mutator = TeleportEntityToIllegalTileMutator(seed=42)
    assert not mutator.precondition("noop", initial_state_obj)
