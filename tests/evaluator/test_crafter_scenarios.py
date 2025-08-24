"""
Tests for Crafter scenarios that verify actual behavior and outcomes.
"""

from crafter.functional_env import transition
from crafter import constants
import pytest

from distant_sunburn.evaluator.crafter.scenarios import (
    CraftWoodenPickaxeScenario,
    CowMovementScenario,
)
from crafter.state_export import Position
from distant_sunburn.evaluator.crafter.components import RandomMovementPolicy


def get_action_index(action: str) -> int:
    """Convert action string to action index."""
    return constants.actions.index(action)  # type: ignore


class TestCraftWoodenPickaxeScenario:
    """Test the wooden pickaxe crafting scenario."""

    def test_scenario_creates_wooden_pickaxe(self):
        """Test that running the scenario results in a wooden pickaxe being crafted."""
        # Arrange
        scenario = CraftWoodenPickaxeScenario()

        # Act - Get initial state and run the scenario
        initial_state = scenario.get_initial_state()
        actions = scenario.get_actions()

        # Verify initial state doesn't have pickaxe
        assert (
            initial_state.player.inventory.wood_pickaxe == 0
        ), "Should not have pickaxe initially"

        # Execute the actions
        current_state = initial_state
        for action in actions:
            action_index = constants.actions.index(action)
            current_state, _ = transition(current_state, action_index)

        # Assert - Verify we now have a wooden pickaxe
        assert (
            current_state.player.inventory.wood_pickaxe == 1
        ), "Should have crafted a wooden pickaxe"


class TestCowMovementScenario:
    """Test the cow movement scenario."""

    def test_scenario_has_cow_in_world(self):
        """Test that the scenario creates a world with a cow present."""
        # Arrange
        scenario = CowMovementScenario()

        # Act
        initial_state = scenario.get_initial_state()

        # Assert - Verify there's a cow in the world
        cows = [obj for obj in initial_state.objects if obj.name == "cow"]
        assert len(cows) == 1

        # Get the initial position of the cow
        cow_position = cows[0].position
        cow_id = cows[0].entity_id

        actions = scenario.get_actions()

        # Execute the actions
        current_state = initial_state
        for action in actions:
            action_index = constants.actions.index(action)
            current_state, _ = transition(current_state, action_index)

            # break if the cow has moved
            cows = [
                obj
                for obj in current_state.objects
                if obj.name == "cow" and obj.entity_id == cow_id
            ]

            if cows and cows[0].position != cow_position:
                # Test passed
                break
        else:
            # Fail the test if the cow didn't move at all
            pytest.fail(f"Cow did not move in {len(actions)} actions")


def test_random_movement_scenario():
    scenario = RandomMovementPolicy(policy_seed=1, num_transitions=100)

    transitions = scenario()

    # Assert that the player moved from the initial position
    initial_position = transitions[0].prev_metadata.player.position

    for t in transitions:
        if t.prev_metadata.player.position != initial_position:
            # Test passed
            break
    else:
        # Fail the test if the player didn't move at all
        pytest.fail(f"Player did not move in {len(transitions)} actions")
