from __future__ import annotations

import numpy as np
import torch

from onelife.duc_wm import (
    SimFuturesTrainerConfig,
    SimFuturesWorldModel,
    SimFuturesWorldModelConfig,
    default_mujoco_templates,
    fit_simfutures_world_model,
)
from onelife.mujoco_dataset import MuJoCoTransitions


def _tiny_transitions() -> MuJoCoTransitions:
    rng = np.random.default_rng(7)
    states = rng.normal(size=(32, 6)).astype(np.float32)
    actions = rng.normal(size=(32, 2)).astype(np.float32)
    delta = np.zeros_like(states)
    delta[:, 3:5] = 0.08 * actions
    delta += 0.01 * rng.normal(size=states.shape).astype(np.float32)
    next_states = states + delta
    rewards = (next_states[:, 3] - 0.01 * np.square(actions).sum(axis=1)).astype(np.float32)
    return MuJoCoTransitions(
        states=states,
        actions=actions,
        next_states=next_states.astype(np.float32),
        rewards=rewards,
        dones=np.zeros(32, dtype=np.bool_),
    )


def test_simfutures_forward_shapes() -> None:
    transitions = _tiny_transitions()
    templates = default_mujoco_templates("Swimmer-v5", transitions.state_dim, transitions.action_dim)
    model = SimFuturesWorldModel(
        SimFuturesWorldModelConfig(
            state_dim=transitions.state_dim,
            action_dim=transitions.action_dim,
            templates=templates,
            hidden_size=16,
            hidden_layers=1,
            history_length=2,
        )
    )
    states = torch.tensor(transitions.states[:4])
    actions = torch.tensor(transitions.actions[:4])
    next_states = torch.tensor(transitions.next_states[:4])
    output = model(states, actions, next_states=next_states, sample_context=False)
    assert output.mean.shape == states.shape
    assert output.logvar.shape == states.shape
    assert output.alpha_mean.shape == (4, len(templates))
    assert output.law_channel_pred.shape == (4, len(templates))
    assert output.planning_bonus.shape == (4,)


def test_simfutures_training_updates_history() -> None:
    transitions = _tiny_transitions()
    templates = default_mujoco_templates("Swimmer-v5", transitions.state_dim, transitions.action_dim)
    model = SimFuturesWorldModel(
        SimFuturesWorldModelConfig(
            state_dim=transitions.state_dim,
            action_dim=transitions.action_dim,
            templates=templates,
            hidden_size=16,
            hidden_layers=1,
            history_length=2,
        )
    )
    history = fit_simfutures_world_model(
        model=model,
        transitions=transitions,
        config=SimFuturesTrainerConfig(
            epochs=1,
            batch_size=8,
            history_length=2,
            posterior_update_samples=16,
            rollout_horizon=1,
            seed=3,
        ),
        device=torch.device("cpu"),
    )
    assert len(history) == 1
    assert "law_channel" in history[0]
    assert "posterior_entropy" in history[0]
