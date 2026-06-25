from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class WorldModelForwardOutput:
    mean: torch.Tensor
    prediction_mean: torch.Tensor
    planning_mean: torch.Tensor
    logvar: torch.Tensor
    effects: torch.Tensor
    prior_effects: torch.Tensor
    residual_effects: torch.Tensor
    raw_prior_effects: torch.Tensor
    raw_residual_effects: torch.Tensor
    alpha: torch.Tensor
    alpha_mean: torch.Tensor
    posterior_mean: torch.Tensor
    posterior_logvar: torch.Tensor
    base_delta: torch.Tensor
    context_delta: torch.Tensor
    prior_delta: torch.Tensor
    residual_delta: torch.Tensor
    mechanism_delta: torch.Tensor
    proposed_mechanism_delta: torch.Tensor
    mechanism_mix: torch.Tensor
    planning_delta: torch.Tensor
    prior_beta: torch.Tensor
    residual_scale: torch.Tensor
    prior_gate: torch.Tensor
    data_confidence: torch.Tensor
    reward_pred: torch.Tensor
    planning_reward_pred: torch.Tensor


def mlp(input_dim: int, output_dim: int, hidden_size: int, hidden_layers: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = input_dim
    for _ in range(hidden_layers):
        layers.append(nn.Linear(last, hidden_size))
        layers.append(nn.SiLU())
        last = hidden_size
    layers.append(nn.Linear(last, output_dim))
    return nn.Sequential(*layers)


def kl_normal_diag(
    mean: torch.Tensor,
    logvar: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_std: torch.Tensor,
) -> torch.Tensor:
    prior_var = prior_std.pow(2).clamp_min(1e-8)
    var = torch.exp(logvar)
    kl = 0.5 * (
        (var + (mean - prior_mean).pow(2)) / prior_var
        - 1.0
        + torch.log(prior_var)
        - logvar
    )
    return kl.sum(dim=-1).mean()


def weighted_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor | None,
) -> torch.Tensor:
    error = (prediction - target).pow(2)
    if weights is not None:
        error = error * weights
    return error.mean()
