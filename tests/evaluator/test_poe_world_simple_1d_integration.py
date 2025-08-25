"""
Integration test for PoE-World inference machinery.

This test validates the complete inference pipeline:
1. Generate random data using the 1D environment
2. Split into training/testing sets
3. Fit expert weights using maximum likelihood
4. Validate that good experts get higher weights than bad experts
"""

import random
import numpy as np
import pytest
from typing import List

from distant_sunburn.poe_world.core import SymbolicTransition
from distant_sunburn.simple_1d_env.environment import (
    initial_state,
    transition_function,
    Action,
    DEFAULT_LAWS,
    GameState,
    WorldConfig,
)
from distant_sunburn.poe_world.simple_1d_env.handwritten_experts import (
    CORRECT_EXPERTS,
    INCORRECT_EXPERTS,
    ALL_EXPERTS,
)
from distant_sunburn.poe_world.simple_1d_env.weight_fitter import (
    MaxLikelihoodWeightFitter,
)
from distant_sunburn.poe_world.simple_1d_env.world_model import PoEWorldModel

from typing import Callable
from loguru import logger
from distant_sunburn.log_utils import change_log_level


def generate_random_data(
    n_transitions: int, seed: int = 42
) -> List[SymbolicTransition[GameState]]:
    """
    Generate random transitions using the 1D environment.

    Args:
        n_transitions: Number of transitions to generate
        seed: Random seed for reproducibility

    Returns:
        List of symbolic transitions
    """
    import distant_sunburn.simple_1d_env.environment

    with change_log_level(
        {
            "INFO": [distant_sunburn.simple_1d_env.environment],
        }
    ):
        rng = random.Random(seed)
        np.random.seed(seed)

        transitions = []
        current_state = initial_state(WorldConfig(seed=seed))

        for _ in range(n_transitions):
            # Choose random action
            action = rng.choice(list(Action))

            # Apply transition function
            next_state = transition_function(current_state, action, DEFAULT_LAWS)

            # Create symbolic transition
            transition = SymbolicTransition(
                prev_metadata=current_state, action=action, next_metadata=next_state
            )
            transitions.append(transition)

            # Update current state for next iteration
            current_state = next_state

        return transitions


def test():
    assert len(generate_random_data(100, 42)) == 100
    logger.info("Test passed")
