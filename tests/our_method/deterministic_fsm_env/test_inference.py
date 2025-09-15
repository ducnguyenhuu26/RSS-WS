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
from distant_sunburn.our_method.deterministic_fsm_env.observable_extractor import (
    ObservableExtractor,
)
from distant_sunburn.our_method.deterministic_fsm_env.handwritten_laws import (
    CorrectToggleALaw,
    IncorrectToggleALawStaysSame,
)
from distant_sunburn.our_method.core import WeightedLaw, LawFunctionWrapper
from distant_sunburn.our_method.world_modeling import LawMixture
from collections import Counter
import rich


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


def test_preconditions_checked(monkeypatch):
    weighted_laws = [
        WeightedLaw(law=fn, weight=1.0, is_fitted=True) for fn in CORRECT_LAWS
    ]
    world_model = LawMixture(
        observable_extractor=ObservableExtractor(),
        weighted_laws=weighted_laws,
    )

    # Track precondition calls
    precondition_calls = []

    # Apply spy to each law's precondition method
    for weighted_law in weighted_laws:
        original_precondition = weighted_law.law.precondition

        def make_spy_wrapper(law, original_method):
            def spy_wrapper(current_state, action):
                precondition_calls.append(
                    {
                        "law_name": law.__class__.__name__,
                        "state": current_state,
                        "action": action,
                    }
                )
                return original_method(current_state, action)

            return spy_wrapper

        monkeypatch.setattr(
            weighted_law.law,
            "precondition",
            make_spy_wrapper(weighted_law.law, original_precondition),
        )

    for action in Action:
        state = initial_state()
        world_model.sample_next_state(state, action)

    # Verify that preconditions were called
    assert len(precondition_calls) > 0, "No precondition calls were made"

    # Verify that each law's precondition was called for each action
    expected_calls = len(CORRECT_LAWS) * len(Action)
    assert (
        len(precondition_calls) == expected_calls
    ), f"Expected {expected_calls} precondition calls, got {len(precondition_calls)}"

    # Verify that all laws were called
    called_laws = {call["law_name"] for call in precondition_calls}
    expected_law_names = {law.__class__.__name__ for law in CORRECT_LAWS}
    assert (
        called_laws == expected_law_names
    ), f"Expected laws {expected_law_names}, but got {called_laws}"

    # Verify that all actions were tested
    called_actions = {call["action"] for call in precondition_calls}
    expected_actions = set(Action)
    assert (
        called_actions == expected_actions
    ), f"Expected actions {expected_actions}, but got {called_actions}"


def test_no_implicit_uniform_on_unobserved_attributes():
    weighted_laws = [
        WeightedLaw(law=fn, weight=1.0, is_fitted=True) for fn in CORRECT_LAWS
    ]
    world_model = LawMixture(
        observable_extractor=ObservableExtractor(),
        weighted_laws=weighted_laws,
    )

    n_samples = 100

    SWITCH_A_INITIAL_VALUE = 0
    SWITCH_B_INITIAL_VALUE = 1

    switch_a_values: list[int] = []
    switch_b_values: list[int] = []
    for _ in range(n_samples):
        state = State(switch_a=SWITCH_A_INITIAL_VALUE, switch_b=SWITCH_B_INITIAL_VALUE)
        action = Action.TOGGLE_A
        next_state = world_model.sample_next_state(state, action)
        switch_a_values.append(next_state.switch_a)
        switch_b_values.append(next_state.switch_b)

    switch_a_counts = Counter(switch_a_values)
    switch_b_counts = Counter(switch_b_values)

    switch_a_posterior = np.zeros(2)
    switch_b_posterior = np.zeros(2)

    for k, v in switch_a_counts.items():
        switch_a_posterior[k] = v / len(switch_a_values)
    for k, v in switch_b_counts.items():
        switch_b_posterior[k] = v / len(switch_b_values)

    rich.print(
        {
            "Switch A posterior": switch_a_posterior,
            "Switch B posterior": switch_b_posterior,
        }
    )

    # Switch A should get toggled away from its initial value
    assert switch_a_posterior[SWITCH_A_INITIAL_VALUE] == 0
    assert switch_a_posterior[int(not SWITCH_A_INITIAL_VALUE)] == 1

    # We never took a TOGGLE_B action, so switch b should never get toggled
    # away from its initial value
    assert switch_b_posterior[SWITCH_B_INITIAL_VALUE] == 1
    assert switch_b_posterior[int(not SWITCH_B_INITIAL_VALUE)] == 0


# def test_jumpy_posterior():
#     weighted_laws = [
#         WeightedLaw(law=fn, weight=1.0, is_fitted=True) for fn in CORRECT_LAWS
#     ]
#     world_model = LawMixture(
#         observable_extractor=ObservableExtractor(),
#         weighted_laws=weighted_laws,
#     )

#     n_samples = 100

#     switch_a_values: list[int] = []
#     switch_b_values: list[int] = []
#     for _ in range(n_samples):
#         state = State(switch_a=0, switch_b=1)
#         action = Action.TOGGLE_A
#         next_state = world_model.sample_next_state(state, action)
#         switch_a_values.append(next_state.switch_a)
#         switch_b_values.append(next_state.switch_b)

#     for _ in range(n_samples):
#         state = State(switch_a=0, switch_b=1)
#         action = Action.TOGGLE_B
#         next_state = world_model.sample_next_state(state, action)
#         switch_a_values.append(next_state.switch_a)
#         switch_b_values.append(next_state.switch_b)

#     switch_a_counts = Counter(switch_a_values)
#     switch_b_counts = Counter(switch_b_values)

#     switch_a_posterior = {
#         k: v / len(switch_a_values) for k, v in switch_a_counts.items()
#     }

#     switch_b_posterior = {k: v / n_samples for k, v in switch_b_counts.items()}

#     rich.print(
#         {
#             "Switch A posterior": switch_a_posterior,
#             "Switch B posterior": switch_b_posterior,
#         }
#     )

#     concentration = 0.8
#     for name, posterior in [
#         ("Switch A", switch_a_posterior),
#         ("Switch B", switch_b_posterior),
#     ]:
#         for k, v in posterior.items():
#             assert v <= concentration, f"{name} P({k}) = {v} > {concentration}"
