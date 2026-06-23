from __future__ import annotations

import numpy as np
import torch

from onelife.duc_wm import (
    DUCTrainerConfig,
    DUCWorldModel,
    DUCWorldModelConfig,
    default_mujoco_templates,
    evaluate_duc_model,
    fit_duc_world_model,
)
from onelife.mujoco_dataset import MuJoCoTransitions


def test_duc_world_model_forward_shapes():
    templates = default_mujoco_templates("Ant-v5", state_dim=8, action_dim=3)
    model = DUCWorldModel(
        DUCWorldModelConfig(
            state_dim=8,
            action_dim=3,
            templates=templates,
            hidden_size=16,
            hidden_layers=1,
            history_length=2,
        )
    )
    states = torch.zeros(4, 8)
    actions = torch.zeros(4, 3)
    history_states = torch.zeros(4, 2, 8)
    history_actions = torch.zeros(4, 2, 3)

    output = model(states, actions, history_states, history_actions)

    assert output.mean.shape == (4, 8)
    assert output.logvar.shape == (4, 8)
    assert output.effects.shape == (4, len(templates), 8)
    assert output.alpha.shape == (4, len(templates))


def test_duc_training_smoke_on_synthetic_contexts():
    transitions = MuJoCoTransitions(
        states=np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 1.0], [3.0, 1.5]], dtype=np.float32),
        actions=np.array([[1.0], [1.0], [1.0], [1.0]], dtype=np.float32),
        next_states=np.array([[1.0, 0.5], [2.0, 1.0], [3.0, 1.5], [4.0, 2.0]], dtype=np.float32),
        contexts=np.array([[0.2], [0.2], [0.2], [0.2]], dtype=np.float32),
        context_names=("actuation",),
    )
    templates = default_mujoco_templates("Tiny-v0", state_dim=2, action_dim=1)[:1]
    model = DUCWorldModel(
        DUCWorldModelConfig(
            state_dim=2,
            action_dim=1,
            templates=templates,
            hidden_size=16,
            hidden_layers=1,
            history_length=2,
        )
    )

    history = fit_duc_world_model(
        model,
        transitions,
        DUCTrainerConfig(epochs=1, batch_size=2, history_length=2),
        device="cpu",
    )
    metrics = evaluate_duc_model(model, transitions, device="cpu", batch_size=2, history_length=2)

    assert history
    assert "r2_at_1" in metrics
    assert "duc_r2_at_1" in metrics
