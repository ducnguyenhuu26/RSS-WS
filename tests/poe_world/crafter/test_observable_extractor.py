"""
Tests for the Crafter ObservableExtractor.
"""

import pytest
import torch
import numpy as np

from distant_sunburn.poe_world.crafter.observable_extractor import ObservableExtractor
from distant_sunburn.poe_world.core import ObservableId, DiscreteDistribution
from crafter.functional_env import initial_state


class TestObservableExtractor:
    """Test the Crafter ObservableExtractor implementation."""

    def test_extract_attribute_predictions(self):
        """Test extracting attribute predictions from a Crafter state."""
        # Create initial state
        state = initial_state()

        # Create extractor
        extractor = ObservableExtractor()

        # Extract predictions
        predictions = extractor.extract_attribute_predictions(state)

        # Check that we get predictions for player attributes
        assert ObservableId("player_position_x") in predictions
        assert ObservableId("player_position_y") in predictions
        assert ObservableId("player_health") in predictions

        # Check that predictions are DiscreteDistribution objects
        assert isinstance(
            predictions[ObservableId("player_position_x")], DiscreteDistribution
        )
        assert isinstance(
            predictions[ObservableId("player_position_y")], DiscreteDistribution
        )
        assert isinstance(
            predictions[ObservableId("player_health")], DiscreteDistribution
        )

        # Check that domains match the extractor's defined domains
        assert np.array_equal(
            predictions[ObservableId("player_position_x")].support,
            extractor.position_domain,
        )
        assert np.array_equal(
            predictions[ObservableId("player_position_y")].support,
            extractor.position_domain,
        )
        assert np.array_equal(
            predictions[ObservableId("player_health")].support, extractor.health_domain
        )

    def test_get_observed_outcomes(self):
        """Test extracting observed outcomes from a Crafter state."""
        # Create initial state
        state = initial_state()

        # Create extractor
        extractor = ObservableExtractor()

        # Extract observed outcomes
        observed = extractor.get_observed_outcomes(state)

        # Check that we get observed values for player attributes
        assert ObservableId("player_position_x") in observed
        assert ObservableId("player_position_y") in observed
        assert ObservableId("player_health") in observed

        # Check that observed values are numeric and within expected ranges
        assert observed[ObservableId("player_position_x")] >= 0
        assert observed[ObservableId("player_position_x")] < len(
            extractor.position_domain
        )
        assert observed[ObservableId("player_position_y")] >= 0
        assert observed[ObservableId("player_position_y")] < len(
            extractor.position_domain
        )
        assert observed[ObservableId("player_health")] >= 0
        assert observed[ObservableId("player_health")] < len(extractor.health_domain)

        # Check that values match the actual state
        assert observed[ObservableId("player_position_x")] == state.player.position.x
        assert observed[ObservableId("player_position_y")] == state.player.position.y
        assert observed[ObservableId("player_health")] == state.player.health

    def test_apply_expert_predictions(self):
        """Test applying expert predictions to create a new state."""
        # Create initial state
        state = initial_state()

        # Create extractor
        extractor = ObservableExtractor()

        # Create mock expert predictions
        expert_predictions = {
            ObservableId("player_position_x"): [
                DiscreteDistribution.from_uniform(extractor.position_domain),
                DiscreteDistribution.from_uniform(extractor.position_domain),
            ],
            ObservableId("player_position_y"): [
                DiscreteDistribution.from_uniform(extractor.position_domain),
                DiscreteDistribution.from_uniform(extractor.position_domain),
            ],
            ObservableId("player_health"): [
                DiscreteDistribution.from_uniform(extractor.health_domain),
                DiscreteDistribution.from_uniform(extractor.health_domain),
            ],
        }

        # Create weights tensor
        weights = torch.tensor([0.5, 0.5], dtype=torch.float32)

        # Apply expert predictions (modifies state in-place)
        extractor.apply_expert_predictions(state, expert_predictions, weights)

        # Check that player attributes were sampled and are within valid ranges
        assert state.player.position.x >= 0
        assert state.player.position.x < len(extractor.position_domain)
        assert state.player.position.y >= 0
        assert state.player.position.y < len(extractor.position_domain)
        assert state.player.health >= 0
        assert state.player.health < len(extractor.health_domain)
