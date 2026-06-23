from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import numpy as np
import torch

from .model import DUCWorldModel


RewardFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class CEMPlannerConfig:
    horizon: int = 20
    candidates: int = 1024
    elites: int = 128
    iterations: int = 4
    context_samples: int = 4
    uncertainty_weight: float = 0.05


@torch.no_grad()
def plan_cem_action(
    model: DUCWorldModel,
    state: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
    reward_fn: RewardFn,
    config: CEMPlannerConfig,
    device: torch.device | str,
) -> np.ndarray:
    """Return the first MPC action from a compact CEM planner."""

    model.eval()
    action_dim = int(action_low.shape[0])
    mean = torch.zeros(config.horizon, action_dim, device=device)
    std = torch.ones_like(mean)
    low = torch.tensor(action_low, dtype=torch.float32, device=device)
    high = torch.tensor(action_high, dtype=torch.float32, device=device)
    initial_state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for _ in range(config.iterations):
        samples = mean + std * torch.randn(config.candidates, config.horizon, action_dim, device=device)
        samples = torch.max(torch.min(samples, high), low)
        returns = rollout_action_sequences(model, initial_state, samples, reward_fn, config)
        elite_indices = returns.topk(k=min(config.elites, config.candidates)).indices
        elites = samples.index_select(dim=0, index=elite_indices)
        mean = elites.mean(dim=0)
        std = elites.std(dim=0).clamp_min(1e-3)
    return mean[0].cpu().numpy()


def rollout_action_sequences(
    model: DUCWorldModel,
    initial_state: torch.Tensor,
    action_sequences: torch.Tensor,
    reward_fn: RewardFn,
    config: CEMPlannerConfig,
) -> torch.Tensor:
    candidates, horizon, _ = action_sequences.shape
    state = initial_state.expand(candidates, -1)
    total = torch.zeros(candidates, device=action_sequences.device)
    for step in range(horizon):
        action = action_sequences[:, step, :]
        output = model(state, action, context=None, sample_context=False)
        reward = reward_fn(state, action, output.mean)
        uncertainty = torch.exp(output.logvar).mean(dim=-1)
        total = total + reward - config.uncertainty_weight * uncertainty
        state = output.mean
    return total
