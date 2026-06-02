from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


def _activation_module(name: str) -> nn.Module:
    match name:
        case "relu":
            return nn.ReLU()
        case "gelu":
            return nn.GELU()
        case "tanh":
            return nn.Tanh()
        case "silu" | "swish":
            return nn.SiLU()
        case _:
            raise ValueError(f"Unsupported activation: {name}")


class ResidualMLP(nn.Module):
    """MLP that predicts continuous residual dynamics."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (128, 128),
        activation: str = "silu",
        include_unknown_mask: bool = True,
        zero_init_output: bool = True,
    ) -> None:
        super().__init__()
        if state_dim <= 0 or action_dim <= 0:
            raise ValueError("state_dim and action_dim must be positive")
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.include_unknown_mask = bool(include_unknown_mask)

        input_dim = state_dim + action_dim + state_dim
        if include_unknown_mask:
            input_dim += state_dim

        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(previous_dim, int(hidden_size)))
            layers.append(_activation_module(activation))
            previous_dim = int(hidden_size)
        output_layer = nn.Linear(previous_dim, state_dim)
        if zero_init_output:
            nn.init.zeros_(output_layer.weight)
            nn.init.zeros_(output_layer.bias)
        layers.append(output_layer)
        self.net = nn.Sequential(*layers)

    def build_features(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        program_next_states: torch.Tensor,
        unknown_mask: torch.Tensor,
    ) -> torch.Tensor:
        features = [states, actions, program_next_states]
        if self.include_unknown_mask:
            features.append(unknown_mask)
        return torch.cat(features, dim=-1)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        program_next_states: torch.Tensor,
        unknown_mask: torch.Tensor,
    ) -> torch.Tensor:
        features = self.build_features(
            states=states,
            actions=actions,
            program_next_states=program_next_states,
            unknown_mask=unknown_mask,
        )
        return self.net(features)
