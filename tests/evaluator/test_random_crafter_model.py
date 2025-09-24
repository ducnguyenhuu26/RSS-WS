"""Tests for RandomCrafterWorldModel."""

from crafter.constants import ActionT
from crafter.functional_env import initial_state

from distant_sunburn.evaluator.random_crafter_model import RandomCrafterWorldModel


class TestRandomCrafterWorldModel:
    """Test suite for RandomCrafterWorldModel."""

    def test_instantiation(self):
        """Test that the model can be instantiated."""
        model = RandomCrafterWorldModel()
        assert model is not None

    def test_evaluate_log_probability_returns_zero(self):
        """Test that evaluate_log_probability returns 0.0 for random model."""
        model = RandomCrafterWorldModel()
        state = initial_state(area=(9, 9), view=(9, 9))

        result = model.evaluate_log_probability(state, "noop", state)
        assert result == 0.0

    def test_evaluate_log_probability_with_different_actions(self):
        """Test that evaluate_log_probability returns 0.0 for different actions."""
        model = RandomCrafterWorldModel()
        state = initial_state(area=(9, 9), view=(9, 9))
        actions: list[ActionT] = ["noop", "move_left", "do", "sleep"]

        for action in actions:
            result = model.evaluate_log_probability(state, action, state)
            assert result == 0.0, f"Failed for action {action}"

    def test_sample_next_state_produces_different_state(self):
        """Test that sample_next_state produces a state different from the input."""
        model = RandomCrafterWorldModel()
        state = initial_state(area=(9, 9), view=(9, 9))

        next_state = model.sample_next_state(state, "noop")

        # The random model should produce a different state
        assert next_state != state
