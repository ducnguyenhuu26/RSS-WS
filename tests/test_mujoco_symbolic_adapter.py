import math

import numpy as np

from onelife.mujoco_dataset import MuJoCoTransitions
from onelife.mujoco_symbolic_adapter import (
    BinnedMuJoCoState,
    MuJoCoBinnedObservableExtractor,
    MuJoCoDiscretizer,
    make_onelife_mujoco_law_mixture,
    make_poe_mujoco_baseline,
    to_law_symbolic_transitions,
    to_poe_symbolic_transitions,
)


def make_dataset() -> MuJoCoTransitions:
    return MuJoCoTransitions(
        states=np.array(
            [
                [0.0, 0.0],
                [0.5, 0.2],
                [1.0, 0.4],
                [1.5, 0.6],
            ],
            dtype=np.float32,
        ),
        actions=np.array(
            [
                [-1.0],
                [0.0],
                [1.0],
                [0.5],
            ],
            dtype=np.float32,
        ),
        next_states=np.array(
            [
                [0.5, 0.2],
                [1.0, 0.4],
                [1.5, 0.6],
                [2.0, 0.8],
            ],
            dtype=np.float32,
        ),
    )


def test_discretizer_converts_shared_dataset_to_symbolic_transitions():
    dataset = make_dataset()
    discretizer = MuJoCoDiscretizer.fit(dataset, state_bins=5, action_bins=3)

    poe_transitions = to_poe_symbolic_transitions(dataset, discretizer)
    law_transitions = to_law_symbolic_transitions(dataset, discretizer)

    assert len(poe_transitions) == dataset.num_steps
    assert len(law_transitions) == dataset.num_steps
    assert isinstance(poe_transitions[0].prev_metadata, BinnedMuJoCoState)
    assert isinstance(law_transitions[0].prev_state, BinnedMuJoCoState)
    assert len(poe_transitions[0].prev_metadata.bins) == dataset.state_dim


def test_discretizer_undigitizes_binned_state_and_action():
    dataset = make_dataset()
    discretizer = MuJoCoDiscretizer.fit(dataset, state_bins=5, action_bins=3)
    binned_state = discretizer.digitize_state(dataset.states[0])
    binned_action = discretizer.digitize_action(dataset.actions[0])

    continuous_state = discretizer.undigitize_state(binned_state)
    continuous_action = discretizer.undigitize_action(binned_action)

    assert continuous_state.shape == (dataset.state_dim,)
    assert continuous_action.shape == (dataset.action_dim,)
    assert np.isfinite(continuous_state).all()
    assert np.isfinite(continuous_action).all()


def test_poe_world_model_runs_on_binned_mujoco_dataset():
    dataset = make_dataset()
    discretizer = MuJoCoDiscretizer.fit(dataset, state_bins=5, action_bins=3)
    transitions = to_poe_symbolic_transitions(dataset, discretizer)
    model = make_poe_mujoco_baseline(discretizer)

    sampled = model.sample_next_state(
        transitions[0].prev_metadata,
        transitions[0].action,
    )
    log_probability = model.evaluate_log_probability(
        transitions[0].prev_metadata,
        transitions[0].action,
        transitions[0].next_metadata,
    )

    assert isinstance(sampled, BinnedMuJoCoState)
    assert math.isfinite(log_probability)


def test_onelife_law_mixture_runs_on_binned_mujoco_dataset():
    dataset = make_dataset()
    discretizer = MuJoCoDiscretizer.fit(dataset, state_bins=5, action_bins=3)
    transitions = to_law_symbolic_transitions(dataset, discretizer)
    model = make_onelife_mujoco_law_mixture(discretizer)

    sampled = model.sample_next_state(
        transitions[0].prev_state,
        transitions[0].action,
    )
    log_probability = model.evaluate_log_probability(
        transitions[0].prev_state,
        transitions[0].action,
        transitions[0].next_state,
    )

    assert isinstance(sampled, BinnedMuJoCoState)
    assert math.isfinite(log_probability)


def test_observable_extractor_rejects_prediction_state_as_observation():
    dataset = make_dataset()
    discretizer = MuJoCoDiscretizer.fit(dataset, state_bins=5, action_bins=3)
    extractor = MuJoCoBinnedObservableExtractor.from_discretizer(discretizer)
    model = make_onelife_mujoco_law_mixture(discretizer)
    state = discretizer.digitize_state(dataset.states[0])
    model.laws[0].law.effect(state, discretizer.digitize_action(dataset.actions[0]))

    try:
        extractor.get_observed_outcomes(state)
    except ValueError as error:
        assert "predictions" in str(error)
    else:
        raise AssertionError("expected prediction states to be rejected")
