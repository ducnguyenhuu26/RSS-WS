from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn


RewardFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class CEMPlannerConfig:
    horizon: int = 20
    candidates: int = 1024
    elites: int = 128
    iterations: int = 4
    context_samples: int = 4
    uncertainty_weight: float = 0.0
    model_bonus_weight: float = 0.0
    certified_risk_weight: float = 0.0


@torch.no_grad()
def plan_cem_action(
    model: nn.Module,
    state: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
    reward_fn: RewardFn,
    config: CEMPlannerConfig,
    device: torch.device | str,
    history_states: np.ndarray | None = None,
    history_actions: np.ndarray | None = None,
    model_context: np.ndarray | torch.Tensor | None = None,
) -> np.ndarray:
    """Return the first MPC action from a compact CEM planner."""

    model.eval()
    action_dim = int(action_low.shape[0])
    mean = torch.zeros(config.horizon, action_dim, device=device)
    std = torch.ones_like(mean)
    low = torch.tensor(action_low, dtype=torch.float32, device=device)
    high = torch.tensor(action_high, dtype=torch.float32, device=device)
    initial_state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    history_state_tensor = None
    history_action_tensor = None
    if history_states is not None and history_actions is not None:
        history_state_tensor = torch.tensor(
            history_states,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        history_action_tensor = torch.tensor(
            history_actions,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
    context_tensor = None
    if model_context is not None:
        context_tensor = torch.as_tensor(
            model_context,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
    for _ in range(config.iterations):
        samples = mean + std * torch.randn(config.candidates, config.horizon, action_dim, device=device)
        samples = torch.max(torch.min(samples, high), low)
        returns = rollout_action_sequences(
            model,
            initial_state,
            samples,
            reward_fn,
            config,
            history_states=history_state_tensor,
            history_actions=history_action_tensor,
            model_context=context_tensor,
        )
        elite_indices = returns.topk(k=min(config.elites, config.candidates)).indices
        elites = samples.index_select(dim=0, index=elite_indices)
        mean = elites.mean(dim=0)
        std = elites.std(dim=0).clamp_min(1e-3)
    return mean[0].cpu().numpy()


def rollout_action_sequences(
    model: nn.Module,
    initial_state: torch.Tensor,
    action_sequences: torch.Tensor,
    reward_fn: RewardFn,
    config: CEMPlannerConfig,
    history_states: torch.Tensor | None = None,
    history_actions: torch.Tensor | None = None,
    model_context: torch.Tensor | None = None,
) -> torch.Tensor:
    candidates, horizon, _ = action_sequences.shape
    state = initial_state.expand(candidates, -1)
    if history_states is not None:
        history_states = history_states.expand(candidates, -1, -1).clone()
    if history_actions is not None:
        history_actions = history_actions.expand(candidates, -1, -1).clone()
    if model_context is not None:
        model_context = model_context.expand(candidates, -1)
    total = torch.zeros(candidates, device=action_sequences.device)
    use_belief_state = hasattr(model, "initial_belief_state")
    belief_state = None
    for step in range(horizon):
        action = action_sequences[:, step, :]
        if use_belief_state:
            if belief_state is None:
                belief_state = model.initial_belief_state(
                    state,
                    action,
                    history_states,
                    history_actions,
                )
            output = model(
                state,
                action,
                history_states,
                history_actions,
                context=model_context,
                sample_context=False,
                belief_state=belief_state,
            )
            belief_state = getattr(output, "belief_next", belief_state)
        else:
            output = model(
                state,
                action,
                history_states,
                history_actions,
                context=model_context,
                sample_context=False,
            )
        reward = reward_fn(state, action, output.mean)
        uncertainty = torch.exp(output.logvar).mean(dim=-1)
        bonus = getattr(output, "planning_bonus", None)
        if bonus is None:
            bonus = torch.zeros_like(reward)
        risk = getattr(output, "certified_risk", None)
        if risk is None:
            risk = torch.zeros_like(reward)
        risk_scale = getattr(model, "certified_risk_scale", None)
        if torch.is_tensor(risk_scale):
            risk_scale_value = float(risk_scale.detach().cpu())
        elif risk_scale is None:
            risk_scale_value = 1.0
        else:
            risk_scale_value = float(risk_scale)
        total = (
            total
            + reward
            + float(config.model_bonus_weight) * bonus
            - float(config.certified_risk_weight) * risk_scale_value * risk
            - config.uncertainty_weight * uncertainty
        )
        state = output.mean
        if history_states is not None:
            history_states = torch.cat([history_states[:, 1:], state.unsqueeze(1)], dim=1)
        if history_actions is not None:
            history_actions = torch.cat([history_actions[:, 1:], action.unsqueeze(1)], dim=1)
    return total
