"""
Tests for the handwritten experts for the Crafter environment.

This module tests that the correct experts model the right mechanics
and that the incorrect experts make obviously wrong predictions.
"""

import pytest
from crafter.state_export import WorldState, Position, PlayerState, CowState
from crafter.functional_env import initial_state
from crafter.constants import ActionT

from distant_sunburn.poe_world.crafter.handwritten_experts import (
    correct_player_movement_expert,
    correct_combat_damage_expert,
    correct_entity_ai_expert,
    incorrect_player_movement_expert_teleports,
    incorrect_combat_damage_expert_instakills,
    incorrect_entity_ai_expert_self_destructs,
)
from distant_sunburn.poe_world.core import DiscreteDistribution


def create_simple_test_state() -> WorldState:
    """Create a simple test state with just a player."""
    return initial_state(area=(5, 5), view=(3, 3), seed=42)


class TestCorrectExperts:
    """Test that correct experts model the right mechanics."""

    def test_correct_movement_expert_moves_player(self):
        """Test that movement expert correctly predicts player movement."""
        # Arrange
        state = create_simple_test_state()
        initial_x = state.player.position.x
        initial_y = state.player.position.y

        # Act
        correct_player_movement_expert(state, "move_right")

        # Assert - Should predict moving one step right
        assert isinstance(state.player.position.x, DiscreteDistribution)
        assert state.player.position.x.support[0] == initial_x + 1
        assert state.player.position.y.support[0] == initial_y

    def test_correct_movement_expert_respects_boundaries(self):
        """Test that movement expert respects world boundaries."""
        # Arrange - Player at right edge
        state = create_simple_test_state()
        state.player.position.x = 4  # Right edge of 5x5 world
        state.player.position.y = 2

        # Act
        correct_player_movement_expert(state, "move_right")

        # Assert - Should stay at edge
        assert isinstance(state.player.position.x, DiscreteDistribution)
        assert state.player.position.x.support[0] == 4  # Should not go beyond boundary

    def test_correct_combat_expert_damages_entity(self):
        """Test that combat expert correctly predicts damage to entities."""
        # Arrange
        state = create_simple_test_state()

        # Add a cow in front of player
        cow = CowState(
            entity_id=2,
            position=Position(x=3, y=2),  # Right in front of player at (2,2)
            health=5,
            name="cow",
        )
        state.objects.append(cow)
        state.player.facing = Position(x=1, y=0)  # Facing right
        state.player.inventory.wood_sword = 1  # Has weapon

        # Act
        correct_combat_damage_expert(state, "do")

        # Assert - Should predict damage to cow
        assert isinstance(cow.health, DiscreteDistribution)
        assert cow.health.support[0] == 3  # 5 - 2 (wood sword damage)


class TestIncorrectExperts:
    """Test that incorrect experts make obviously wrong predictions."""

    def test_incorrect_movement_expert_teleports(self):
        """Test that incorrect movement expert teleports instead of moving."""
        # Arrange
        state = create_simple_test_state()
        initial_x = state.player.position.x
        initial_y = state.player.position.y

        # Act
        incorrect_player_movement_expert_teleports(state, "move_right")

        # Assert - Should teleport to random position, not move one step
        assert isinstance(state.player.position.x, DiscreteDistribution)
        assert isinstance(state.player.position.y, DiscreteDistribution)

        teleported_x = state.player.position.x.support[0]
        teleported_y = state.player.position.y.support[0]

        # Should not be the expected position (initial_x + 1, initial_y)
        assert not (teleported_x == initial_x + 1 and teleported_y == initial_y)

    def test_incorrect_combat_expert_instakills(self):
        """Test that incorrect combat expert instantly kills entities."""
        # Arrange
        state = create_simple_test_state()
        cow = CowState(entity_id=2, position=Position(x=3, y=2), health=5, name="cow")
        state.objects.append(cow)
        state.player.facing = Position(x=1, y=0)

        # Act
        incorrect_combat_damage_expert_instakills(state, "do")

        # Assert - Should instantly kill the entity
        assert isinstance(cow.health, DiscreteDistribution)
        assert cow.health.support[0] == 0

    def test_incorrect_entity_expert_self_destructs(self):
        """Test that incorrect entity expert makes all entities self-destruct."""
        # Arrange
        state = create_simple_test_state()
        cow = CowState(entity_id=2, position=Position(x=3, y=2), health=5, name="cow")
        state.objects.append(cow)

        # Act
        incorrect_entity_ai_expert_self_destructs(state, "move_right")

        # Assert - All entities should self-destruct
        assert isinstance(cow.health, DiscreteDistribution)
        assert cow.health.support[0] == 0

        # Player should not be affected
        assert not isinstance(state.player.health, DiscreteDistribution)
