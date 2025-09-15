import random
import numpy as np
import pytest
from typing import List

from distant_sunburn.our_method.core import SymbolicTransition
from distant_sunburn.deterministic_fsm_env import (
    initial_state,
    transition_function,
    Action,
    State,
)
from distant_sunburn.our_method.deterministic_fsm_env.handwritten_laws import (
    CORRECT_LAWS,
    INCORRECT_LAWS,
    ALL_LAWS,
)
from distant_sunburn.our_method.optimization import (
    MaxLikelihoodWeightFitter,
)
from distant_sunburn.poe_world.deterministic_fsm_env.observable_extractor import (
    ObservableExtractor,
)
from distant_sunburn.our_method.deterministic_fsm_env.handwritten_laws import (
    CorrectToggleALaw,
    IncorrectToggleALawStaysSame,
)
from distant_sunburn.our_method.core import WeightedLaw, LawFunctionWrapper
from distant_sunburn.our_method.world_modeling import LawMixture


def generate_random_data(
    n_transitions: int, seed: int = 42
) -> List[SymbolicTransition[State]]:
    """
    Generate random transitions using the 1D environment.

    Args:
        n_transitions: Number of transitions to generate
        seed: Random seed for reproducibility

    Returns:
        List of symbolic transitions
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    transitions: List[SymbolicTransition[State]] = []
    current_state = initial_state()

    for _ in range(n_transitions):
        # Choose random action
        action = rng.choice(list(Action))

        # Apply transition function
        next_state = transition_function(current_state, action)

        # Create symbolic transition
        transition = SymbolicTransition(
            prev_state=current_state, action=action, next_state=next_state
        )
        transitions.append(transition)

        # Update current state for next iteration
        current_state = next_state

    return transitions


def test_weight_fitting_distinguishes_single_good_from_bad_expert():
    n_transitions = 50
    transitions = generate_random_data(n_transitions, seed=42)

    # Split into train/test (75/25)
    split_point = int(0.75 * len(transitions))
    train_transitions = transitions[:split_point]
    test_transitions = transitions[split_point:]

    print(f"Generated {len(transitions)} transitions")
    print(f"Training on {len(train_transitions)} transitions")
    print(f"Testing on {len(test_transitions)} transitions")

    fitter = MaxLikelihoodWeightFitter(
        observable_extractor=ObservableExtractor(),
        learning_rate=0.1,
        max_iterations=5,
        batch_size=200,
        l1_weight=0.001,
    )

    good_law = LawFunctionWrapper[State].from_non_runtime_created(CorrectToggleALaw())
    bad_law = LawFunctionWrapper[State].from_non_runtime_created(
        IncorrectToggleALawStaysSame()
    )

    # Fit weights
    weighted_laws = fitter.fit([good_law, bad_law], train_transitions)

    for weighted_law in weighted_laws:
        print(f"{weighted_law.law.__name__} weight: {weighted_law.weight:.4f}")

    good_weight = weighted_laws[0].weight
    bad_weight = weighted_laws[1].weight

    assert good_weight > bad_weight, (
        f"Good law should have higher weight than bad law. "
        f"Got good={good_weight:.4f}, bad={bad_weight:.4f}"
    )


def test_weight_fitting_distinguishes_good_from_bad_experts():
    n_transitions = 100
    transitions = generate_random_data(n_transitions, seed=42)

    # Split into train/test (75/25)
    split_point = int(0.75 * len(transitions))
    train_transitions = transitions[:split_point]
    test_transitions = transitions[split_point:]

    print(f"Generated {len(transitions)} transitions")
    print(f"Training on {len(train_transitions)} transitions")
    print(f"Testing on {len(test_transitions)} transitions")

    fitter = MaxLikelihoodWeightFitter(
        observable_extractor=ObservableExtractor(),
        learning_rate=0.1,
        max_iterations=5,
        batch_size=200,
        l1_weight=0.001,
    )

    # Fit weights
    weighted_laws = fitter.fit(ALL_LAWS, train_transitions)

    # Extract weights for correct and incorrect experts
    correct_weights = []
    incorrect_weights = []

    for i, weighted_law in enumerate(weighted_laws):
        weight = weighted_law.weight
        law = weighted_law.law

        print(f"Law {i} -- {law.__name__}: weight = {weight:.4f}")

        if law in CORRECT_LAWS:
            correct_weights.append(weight)
            print("  -> CORRECT law")
        elif law in INCORRECT_LAWS:
            incorrect_weights.append(weight)
            print("  -> INCORRECT law")

    # Validate that correct experts have higher average weight
    avg_correct_weight = np.mean(correct_weights)
    avg_incorrect_weight = np.mean(incorrect_weights)

    print(f"\nAverage correct law weight: {avg_correct_weight:.4f}")
    print(f"Average incorrect law weight: {avg_incorrect_weight:.4f}")

    # Main assertion: correct experts should have higher weights on average
    assert avg_correct_weight > avg_incorrect_weight, (
        f"Correct laws should have higher weights than incorrect ones. "
        f"Got correct={avg_correct_weight:.4f}, incorrect={avg_incorrect_weight:.4f}"
    )


def test_jumpy_posterior():
    weighted_laws = [
        WeightedLaw(law=fn, weight=1.0, is_fitted=True) for fn in CORRECT_LAWS
    ]
    world_model = LawMixture(
        observable_extractor=ObservableExtractor(),
        weighted_laws=weighted_laws,
    )

    n_samples = 100

    switch_a_values: list[int] = []
    switch_b_values: list[int] = []
    for _ in range(n_samples):
        state = initial_state()
        action = Action.TOGGLE_A
        next_state = world_model.sample_next_state(state, action)
        switch_a_values.append(next_state.switch_a)
        switch_b_values.append(next_state.switch_b)

    for _ in range(n_samples):
        state = initial_state()
        action = Action.TOGGLE_B
        next_state = world_model.sample_next_state(state, action)
        switch_a_values.append(next_state.switch_a)
        switch_b_values.append(next_state.switch_b)

    # Convert to numpy arrays for cleaner computation
    switch_a_array = np.array(switch_a_values)
    switch_b_array = np.array(switch_b_values)

    # Compute empirical posteriors using numpy
    switch_a_unique, switch_a_counts = np.unique(switch_a_array, return_counts=True)
    switch_b_unique, switch_b_counts = np.unique(switch_b_array, return_counts=True)

    # Convert counts to probabilities
    switch_a_posterior = switch_a_counts / len(switch_a_array)
    switch_b_posterior = switch_b_counts / len(switch_b_array)

    # Assert that no single value has more than max concentration of probability mass
    max_switch_a_prob = np.max(switch_a_posterior)
    max_switch_b_prob = np.max(switch_b_posterior)

    max_concentration = 0.80

    assert (
        max_switch_a_prob <= max_concentration
    ), f"Switch A has max probability {max_switch_a_prob:.3f} > {max_concentration}. "

    assert (
        max_switch_b_prob <= max_concentration
    ), f"Switch B has max probability {max_switch_b_prob:.3f} > {max_concentration}. "

    print(f"Switch A values: {switch_a_unique}")
    print(f"Switch A posterior: {switch_a_posterior}")
    print(f"Switch B values: {switch_b_unique}")
    print(f"Switch B posterior: {switch_b_posterior}")
