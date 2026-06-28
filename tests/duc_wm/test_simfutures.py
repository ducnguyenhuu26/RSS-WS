from __future__ import annotations

import numpy as np
import torch

from onelife.duc_wm import (
    RewardModel,
    RewardModelConfig,
    SimFuturesTrainerConfig,
    SimFuturesWorldModel,
    SimFuturesWorldModelConfig,
    calibrate_certified_risk,
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
    assert output.alpha_dyn_mean.shape == (4, len(templates))
    assert output.alpha_ctrl_mean.shape == (4, len(templates))
    assert output.chart_probs.shape == (4, model.chart_count)
    assert torch.allclose(output.chart_probs.sum(dim=-1), torch.ones(4), atol=1e-5)
    assert output.phase_latent.shape == (4, model.config.phase_dim)
    assert output.phase_next_pred.shape == (4, model.config.phase_dim)
    assert output.phase_next_target.shape == (4, model.config.phase_dim)
    assert output.law_channel_pred.shape == (4, len(templates))
    assert output.planning_bonus.shape == (4,)
    assert output.belief_state.shape == (4, len(templates))
    assert output.belief_next.shape == (4, len(templates))
    assert output.stability_score.shape == (4,)
    assert output.certified_risk.shape == (4,)
    assert output.planning_bonus_gate.shape == (4,)
    assert output.backbone_mean.shape == states.shape
    assert output.backbone_delta.shape == states.shape
    assert output.adapter_delta.shape == states.shape
    assert output.adapter_gate.shape == (4,)
    assert torch.all(output.adapter_gate >= 0.0)
    assert torch.allclose(output.planning_delta, torch.zeros_like(output.planning_delta))


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
    assert "prior_path" in history[0]
    assert "ctrl_kl" in history[0]
    assert "phase" in history[0]
    assert "stability" in history[0]
    assert "risk" in history[0]
    assert "belief_smooth" in history[0]
    assert "backbone" in history[0]
    assert "adapter_safety" in history[0]
    assert "adapter_l1" in history[0]
    assert "rollout_safety" in history[0]
    assert "posterior_entropy" in history[0]


def test_certified_risk_calibration_sets_scale() -> None:
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
    reward_model = RewardModel(
        RewardModelConfig(
            state_dim=transitions.state_dim,
            action_dim=transitions.action_dim,
            hidden_size=16,
            hidden_layers=1,
        )
    )
    stats = calibrate_certified_risk(
        model=model,
        reward_model=reward_model,
        transitions=transitions,
        device=torch.device("cpu"),
        history_length=2,
        batch_size=8,
        max_samples=16,
        delta=0.10,
    )
    assert "certified_risk_scale" in stats
    assert float(model.certified_risk_scale) >= 0.0
    assert 0.0 <= stats["certified_risk_coverage"] <= 1.0
