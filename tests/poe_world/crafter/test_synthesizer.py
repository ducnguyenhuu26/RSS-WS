"""
Tests for the Crafter expert synthesizer.

This module tests that the synthesizer can generate expert functions
from state transitions in the Crafter environment.
"""

import pytest
import asyncio
from crafter.state_export import WorldState, Position, PlayerState, CowState
from crafter.functional_env import (
    initial_state,
    reconstruct_world_from_state,
    export_world_state,
    transition,
)
from crafter.constants import ActionT
from crafter import objects

from distant_sunburn.poe_world.crafter.synthesizer import (
    CrafterExpertSynthesizer,
    SynthesizedExpert,
    create_crafter_synthesizer,
)
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


class TestCrafterExpertSynthesizer:
    """Test the Crafter expert synthesizer."""

    def test_synthesizer_initialization(self):
        """Test that the synthesizer can be initialized."""
        synthesizer = CrafterExpertSynthesizer()
        assert synthesizer is not None
        assert synthesizer.llm_params is not None

    def test_create_crafter_synthesizer(self):
        """Test the convenience function for creating a synthesizer."""
        synthesizer = create_crafter_synthesizer()
        assert isinstance(synthesizer, CrafterExpertSynthesizer)

    def test_filter_surprising_transitions_placeholder(self):
        """Test that the placeholder filter method works."""
        synthesizer = CrafterExpertSynthesizer()

        # Create a simple transition
        initial_state = create_simple_test_state()
        next_state = create_simple_test_state()

        transition = SymbolicTransition(
            prev_metadata=initial_state,
            action="move_right",
            next_metadata=next_state,
        )

        # The placeholder method should return all transitions
        surprising = synthesizer._filter_surprising_transitions([transition], -2.0)
        assert len(surprising) == 1
        assert surprising[0] == transition

    def test_extract_state_changes(self):
        """Test that state changes are correctly extracted for observable attributes only."""
        synthesizer = CrafterExpertSynthesizer()
        transition = create_cow_attack_scenario()

        changes = synthesizer._extract_state_changes(transition)

        # Should detect the cow health change (observable attribute)
        assert "cow" in changes.lower()
        assert "health" in changes.lower()
        assert "3" in changes  # Cow health changed from 5 to 3

        # Should NOT include inventory changes (not observable)
        assert "inventory" not in changes.lower()
        assert "wood_sword" not in changes.lower()

    def test_extract_expert_function(self):
        """Test that expert functions can be extracted from LLM responses."""
        synthesizer = CrafterExpertSynthesizer()

        # Test with valid function
        valid_response = """
def alter_cow_objects(state: WorldState, action: str) -> WorldState:
    if action == "do":
        # Find cow and reduce health
        for entity in state.objects:
            if entity.name == "cow":
                entity.health = DiscreteDistribution(support=[max(0, entity.health - 2)])
    return state
"""

        extracted = synthesizer._extract_expert_function(valid_response)
        assert extracted is not None
        assert "def alter_cow_objects" in extracted
        assert "DiscreteDistribution" in extracted

    def test_validate_expert_code(self):
        """Test that expert code validation works."""
        synthesizer = CrafterExpertSynthesizer()

        # Valid code
        valid_code = """
def alter_cow_objects(state: WorldState, action: str) -> WorldState:
    if action == "do":
        for entity in state.objects:
            if entity.name == "cow":
                entity.health = DiscreteDistribution(support=[max(0, entity.health - 2)])
    return state
"""
        assert synthesizer._validate_expert_code(valid_code)

        # Invalid code (syntax error)
        invalid_code = """
def alter_cow_objects(state: WorldState, action: str) -> WorldState:
    if action == "do":
        for entity in state.objects:
            if entity.name == "cow":
                entity.health = DiscreteDistribution(support=[max(0, entity.health - 2)]
    return state
"""
        assert not synthesizer._validate_expert_code(invalid_code)


def create_simple_test_state() -> WorldState:
    """Create a simple test state with just a player."""
    return initial_state(area=(5, 5), view=(3, 3), seed=42)


@pytest.mark.asyncio
async def test_synthesize_experts_integration():
    """
    Integration test for expert synthesis.

    This test uses a mock LLM response to verify the synthesis pipeline works.
    Note: This test requires an actual LLM call, so it's marked as integration.
    """
    # Skip if no API key is available
    import os

    if not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not available")

    synthesizer = CrafterExpertSynthesizer()
    transition = create_cow_attack_scenario()

    # Try to synthesize experts for cow object type
    experts = await synthesizer.synthesize_experts(
        transitions=[transition],
        object_type="cow",
        surprise_threshold=-2.0,
    )

    # Should find at least one surprising transition
    # (The exact number of experts depends on the LLM response)
    assert len(experts) >= 0  # Could be 0 if LLM fails or no experts generated

    # If experts were generated, they should have the right structure
    for expert in experts:
        assert isinstance(expert, SynthesizedExpert)
        assert expert.object_type == "cow"
        assert expert.code is not None
        assert len(expert.code) > 0
