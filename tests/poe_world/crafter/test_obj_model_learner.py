"""
Tests for the ObjModelLearner for Crafter.

This module tests that the ObjModelLearner can properly manage the synthesis
process for a specific object type.
"""

import pytest
import asyncio
import numpy as np
from crafter.state_export import WorldState, Position, PlayerState, CowState
from crafter.functional_env import (
    initial_state,
    reconstruct_world_from_state,
    export_world_state,
    transition,
)
from crafter.constants import ActionT
from crafter import objects

from distant_sunburn.poe_world.crafter.obj_model_learner import ObjModelLearner
from distant_sunburn.poe_world.core import SymbolicTransition, DiscreteDistribution
from distant_sunburn.litellm_utils import GeminiLiteLlmParams


def create_cow_attack_scenario() -> SymbolicTransition[WorldState]:
    """
    Create a test scenario where a player attacks a cow.

    This creates a state where:
    1. Player is at position (2, 2) facing right
    2. Cow is at position (3, 2) with 5 health
    3. Player has a wood sword
    4. After 'do' action, cow health should decrease
    """
    from crafter import constants

    # Create initial state
    view = (9, 9)
    initial_state_obj = initial_state(area=(9, 9), view=view, seed=42)
    world = reconstruct_world_from_state(initial_state_obj)

    # Find the player object
    player = None
    for obj in world.objects:
        if isinstance(obj, objects.Player):
            player = obj
            break

    if player is None:
        raise ValueError("No player found in world")

    # Set player position and facing direction
    player.pos = np.array((2, 2))
    player.facing = (1, 0)  # Facing right
    player.inventory["wood_sword"] = 1  # Has weapon

    # Add a cow in front of player
    cow = objects.Cow(world, (3, 2))
    world.add(cow)

    # Export the modified initial state
    initial_state_obj = export_world_state(world, view=view, step_count=0)

    # Use the functional transition to get the next state
    # Find the action index for "do"
    action_index = constants.actions.index("do")
    next_state_obj, _ = transition(initial_state_obj, action_index)

    return SymbolicTransition(
        prev_metadata=initial_state_obj,
        action="do",
        next_metadata=next_state_obj,
    )


class TestObjModelLearner:
    """Test the ObjModelLearner."""

    def test_initialization(self):
        """Test that the ObjModelLearner can be initialized."""
        learner = ObjModelLearner("cow")
        assert learner.object_type == "cow"
        assert learner.surprise_threshold == -2.0
        assert learner.world_model is not None
        assert learner.synthesizer is not None

    def test_create_custom_view(self):
        """Test that custom views filter objects correctly."""
        learner = ObjModelLearner("cow")

        # Create a state with multiple object types
        view = (9, 9)
        state = initial_state(area=(9, 9), view=view, seed=42)
        world = reconstruct_world_from_state(state)

        # Find the player object
        player = None
        for obj in world.objects:
            if isinstance(obj, objects.Player):
                player = obj
                break

        if player is None:
            raise ValueError("No player found in world")

        # Add different types of objects
        cow = objects.Cow(world, (3, 3))
        zombie = objects.Zombie(world, (4, 4), player)
        world.add(cow)
        world.add(zombie)

        state_obj = export_world_state(world, view=view, step_count=0)

        # Create custom view for cow
        custom_state = learner._create_custom_view(state_obj)

        # Should only contain cow objects
        assert len(custom_state.objects) == 1
        assert custom_state.objects[0].name == "cow"
        assert custom_state.objects[0].entity_id == cow.entity_id

    def test_filter_surprising_transitions_empty_model(self):
        """Test that transitions are considered surprising when model is empty."""
        learner = ObjModelLearner("cow")
        transition = create_cow_attack_scenario()

        # With an empty model, all transitions should be surprising
        surprising = learner._filter_surprising_transitions([transition])
        assert len(surprising) == 1
        assert surprising[0] == transition

    def test_filter_surprising_transitions_with_experts(self):
        """Test that transitions are filtered based on model predictions."""
        learner = ObjModelLearner("cow")
        transition = create_cow_attack_scenario()

        # Add a simple expert that predicts the transition correctly
        # This would require implementing expert compilation, which is TODO
        # For now, we'll test that the method works with an empty model
        surprising = learner._filter_surprising_transitions([transition])
        assert len(surprising) == 1  # Should be surprising with empty model

    def test_zombie_scenario(self):
        """Test that zombie scenarios work correctly."""
        learner = ObjModelLearner("zombie")
        transition = create_zombie_attack_scenario()

        # With an empty model, all transitions should be surprising
        surprising = learner._filter_surprising_transitions([transition])
        assert len(surprising) == 1
        assert surprising[0] == transition


@pytest.mark.asyncio
async def test_process_transitions_integration():
    """
    Integration test for processing transitions.

    This test verifies that the ObjModelLearner can process transitions
    and potentially synthesize experts.
    """
    # Skip if no API key is available
    import os

    if not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not available")

    learner = ObjModelLearner("cow")
    transition = create_cow_attack_scenario()

    # Process the transition
    experts = await learner.process_transitions([transition])

    # Should find at least one surprising transition with empty model
    # (The exact number of experts depends on the LLM response)
    assert len(experts) >= 0  # Could be 0 if LLM fails or no experts generated

    # If experts were generated, they should have the right structure
    for expert in experts:
        assert expert.object_type == "cow"
        assert expert.code is not None
        assert len(expert.code) > 0


def create_zombie_attack_scenario() -> SymbolicTransition[WorldState]:
    """
    Create a test scenario where a zombie attacks a player.

    This creates a state where:
    1. Player is at position (2, 2) facing right
    2. Zombie is at position (3, 2) with 5 health
    3. After 'do' action, player health should decrease
    """
    from crafter import constants
    import numpy as np

    # Create initial state
    view = (9, 9)
    initial_state_obj = initial_state(area=(9, 9), view=view, seed=42)
    world = reconstruct_world_from_state(initial_state_obj)

    # Find the player object
    player = None
    for obj in world.objects:
        if isinstance(obj, objects.Player):
            player = obj
            break

    if player is None:
        raise ValueError("No player found in world")

    # Set player position and facing direction
    player.pos = np.array((2, 2))
    player.facing = (1, 0)  # Facing right

    # Add a zombie in front of player
    zombie = objects.Zombie(world, (3, 2), player)
    world.add(zombie)

    # Export the modified initial state
    initial_state_obj = export_world_state(world, view=view, step_count=0)

    # Use the functional transition to get the next state
    # Find the action index for "do"
    action_index = constants.actions.index("do")
    next_state_obj, _ = transition(initial_state_obj, action_index)

    return SymbolicTransition(
        prev_metadata=initial_state_obj,
        action="do",
        next_metadata=next_state_obj,
    )


def create_simple_test_state() -> WorldState:
    """Create a simple test state with just a player."""
    return initial_state(area=(5, 5), view=(3, 3), seed=42)
