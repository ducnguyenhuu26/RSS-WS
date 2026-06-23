from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from onelife.mujoco_dataset import MuJoCoTransitions

from .data import iter_duc_batches
from .model import _mlp


@dataclass(frozen=True)
class RewardModelConfig:
    state_dim: int
    action_dim: int
    hidden_size: int = 256
    hidden_layers: int = 2


@dataclass(frozen=True)
class RewardTrainerConfig:
    epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 1e-3
    history_length: int = 1
    seed: int = 0


class RewardModel(nn.Module):
    """Learned reward surrogate shared by all CEM-MPC planners."""

    def __init__(self, config: RewardModelConfig) -> None:
        super().__init__()
        self.config = config
        self.network = _mlp(
            input_dim=2 * config.state_dim + config.action_dim,
            output_dim=1,
            hidden_size=config.hidden_size,
            hidden_layers=config.hidden_layers,
        )

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
    ) -> torch.Tensor:
        inputs = torch.cat([states, actions, next_states], dim=-1)
        return self.network(inputs).squeeze(-1)


def fit_reward_model(
    model: RewardModel,
    transitions: MuJoCoTransitions,
    config: RewardTrainerConfig,
    device: torch.device | str,
) -> list[dict[str, float]]:
    if transitions.rewards is None:
        raise ValueError("reward model training requires transition rewards")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    history: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        losses: list[float] = []
        for batch in iter_duc_batches(
            transitions,
            batch_size=config.batch_size,
            history_length=config.history_length,
            shuffle=True,
            seed=config.seed + epoch,
            device=device,
        ):
            if batch.rewards is None:
                raise ValueError("reward model batch is missing rewards")
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch.states, batch.actions, batch.next_states)
            loss = (prediction - batch.rewards).pow(2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(
            {
                "epoch": float(epoch + 1),
                "loss": sum(losses) / max(1, len(losses)),
            }
        )
    return history
