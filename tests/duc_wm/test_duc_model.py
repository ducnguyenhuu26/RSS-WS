from __future__ import annotations

import numpy as np
import torch

import onelife.duc_wm.trainer as trainer_module
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
    assert output.base_delta.shape == (4, 8)
    assert output.mechanism_delta.shape == (4, 8)
    assert output.reward_pred.shape == (4,)


def test_safe_prior_mixture_starts_as_falsifiable_gate():
    templates = default_mujoco_templates("Ant-v5", state_dim=8, action_dim=3)
    model = DUCWorldModel(
        DUCWorldModelConfig(
            state_dim=8,
            action_dim=3,
            templates=templates,
            hidden_size=16,
            hidden_layers=1,
            history_length=2,
            safe_prior_mixture=True,
            safe_prior_init=0.25,
        )
    )
    states = torch.randn(4, 8)
    actions = torch.randn(4, 3)
    history_states = torch.randn(4, 2, 8)
    history_actions = torch.randn(4, 2, 3)

    output = model(states, actions, history_states, history_actions, sample_context=False)

    assert output.mean.shape == (4, 8)
    assert output.mechanism_mix.shape == (4, 1)
    assert float(output.mechanism_mix.min()) >= 0.0
    assert float(output.mechanism_mix.max()) <= 1.0
    assert torch.allclose(
        output.mechanism_mix.mean(),
        torch.tensor(0.25, dtype=output.mechanism_mix.dtype),
        atol=1e-4,
    )
    assert torch.allclose(
        output.mechanism_delta,
        output.prior_delta + output.residual_delta,
        atol=1e-5,
    )


def test_duc_training_smoke_on_synthetic_contexts():
    transitions = MuJoCoTransitions(
        states=np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 1.0], [3.0, 1.5]], dtype=np.float32),
        actions=np.array([[1.0], [1.0], [1.0], [1.0]], dtype=np.float32),
        next_states=np.array([[1.0, 0.5], [2.0, 1.0], [3.0, 1.5], [4.0, 2.0]], dtype=np.float32),
        contexts=np.array([[0.2], [0.2], [0.2], [0.2]], dtype=np.float32),
        context_names=("actuation",),
    )
    templates = default_mujoco_templates("TinyEnv", state_dim=2, action_dim=1)[:1]
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


def test_prior_validation_can_ignore_context_labels(monkeypatch):
    transitions = MuJoCoTransitions(
        states=np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 1.0]], dtype=np.float32),
        actions=np.array([[1.0], [0.5], [0.0]], dtype=np.float32),
        next_states=np.array([[1.0, 0.5], [1.6, 0.8], [2.1, 1.0]], dtype=np.float32),
        contexts=np.array([[0.2], [0.3], [0.4]], dtype=np.float32),
        context_names=("actuation",),
    )
    templates = default_mujoco_templates("TinyEnv", state_dim=2, action_dim=1)[:1]
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
    seen_contexts = []

    def fake_validate_single_prior(**kwargs):
        seen_contexts.append(kwargs["context"])
        return 0.5, 0.5, 1.0, 0.5

    monkeypatch.setattr(trainer_module, "_validate_single_prior", fake_validate_single_prior)

    trainer_module.calibrate_prior_validation(
        model=model,
        transitions=transitions,
        config=DUCTrainerConfig(
            history_length=2,
            prior_validation_use_context=False,
            context_weight=0.0,
            teacher_force_context=False,
        ),
        device="cpu",
    )

    assert seen_contexts
    assert all(context is None for context in seen_contexts)


def test_duc_reward_wake_replay_training_smoke():
    transitions = MuJoCoTransitions(
        states=np.array(
            [[0.0, 0.0], [1.0, 0.5], [2.0, 1.0], [3.0, 1.5], [4.0, 2.0]],
            dtype=np.float32,
        ),
        actions=np.array([[0.0], [0.5], [1.0], [1.0], [0.0]], dtype=np.float32),
        next_states=np.array(
            [[0.5, 0.1], [1.5, 0.9], [3.0, 1.6], [4.0, 2.1], [4.2, 2.0]],
            dtype=np.float32,
        ),
        rewards=np.array([0.0, 0.5, 2.0, 2.5, 0.1], dtype=np.float32),
        contexts=np.array([[0.1], [0.2], [0.3], [0.3], [0.1]], dtype=np.float32),
        context_names=("actuation",),
    )
    templates = default_mujoco_templates("TinyEnv", state_dim=2, action_dim=1)[:1]
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
        DUCTrainerConfig(
            epochs=1,
            batch_size=2,
            history_length=2,
            reward_weight=0.1,
            wake_replay_weight=0.1,
            action_rank_weight=0.1,
            symbolic_validation_interval=1,
            symbolic_validation_after_training=True,
        ),
        device="cpu",
    )

    assert history
    assert history[-1]["reward"] >= 0.0
    assert history[-1]["wake_replay"] >= 0.0
    assert history[-1]["action_rank"] >= 0.0
    assert "final_prior_gate_mean" in history[-1]
