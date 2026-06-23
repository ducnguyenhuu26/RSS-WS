from __future__ import annotations

import numpy as np

from onelife.duc_wm import (
    BaselineTrainerConfig,
    CaDMWorldModel,
    CaDMWorldModelConfig,
    PETSWorldModel,
    PETSWorldModelConfig,
    default_mujoco_templates,
    evaluate_baseline_world_model,
    fit_baseline_world_model,
)
from onelife.mujoco_dataset import MuJoCoTransitions


def _tiny_transitions() -> MuJoCoTransitions:
    return MuJoCoTransitions(
        states=np.array(
            [[0.0, 0.0], [1.0, 0.5], [2.0, 1.0], [3.0, 1.5]],
            dtype=np.float32,
        ),
        actions=np.array([[1.0], [1.0], [1.0], [1.0]], dtype=np.float32),
        next_states=np.array(
            [[1.0, 0.5], [2.0, 1.0], [3.0, 1.5], [4.0, 2.0]],
            dtype=np.float32,
        ),
    )


def test_pets_baseline_training_and_eval_smoke():
    transitions = _tiny_transitions()
    templates = default_mujoco_templates("TinyEnv", state_dim=2, action_dim=1)
    model = PETSWorldModel(
        PETSWorldModelConfig(
            state_dim=2,
            action_dim=1,
            hidden_size=16,
            hidden_layers=1,
            ensemble_size=2,
        )
    )

    history = fit_baseline_world_model(
        model=model,
        transitions=transitions,
        config=BaselineTrainerConfig(epochs=1, batch_size=2, history_length=2),
        device="cpu",
        control_templates=templates,
    )
    metrics = evaluate_baseline_world_model(
        model=model,
        transitions=transitions,
        device="cpu",
        control_templates=templates,
        batch_size=2,
        history_length=2,
        rollout_horizon=2,
    )

    assert history
    assert "r2_at_1" in metrics
    assert "r2_at_2" in metrics


def test_cadm_baseline_training_and_eval_smoke():
    transitions = _tiny_transitions()
    templates = default_mujoco_templates("TinyEnv", state_dim=2, action_dim=1)
    model = CaDMWorldModel(
        CaDMWorldModelConfig(
            state_dim=2,
            action_dim=1,
            history_length=2,
            context_dim=3,
            hidden_size=16,
            hidden_layers=1,
        )
    )

    history = fit_baseline_world_model(
        model=model,
        transitions=transitions,
        config=BaselineTrainerConfig(epochs=1, batch_size=2, history_length=2),
        device="cpu",
        control_templates=templates,
    )
    metrics = evaluate_baseline_world_model(
        model=model,
        transitions=transitions,
        device="cpu",
        control_templates=templates,
        batch_size=2,
        history_length=2,
        rollout_horizon=2,
    )

    assert history
    assert "r2_at_1" in metrics
    assert "r2_at_2" in metrics
