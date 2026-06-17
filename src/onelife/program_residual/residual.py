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


class ResidualODE(nn.Module):
    """
    Neural ODE residual around the symbolic transition.

    The symbolic program supplies a coarse one-step drift from ``states`` to
    ``program_next_states``. This module keeps that drift fixed over the action
    interval and integrates a learned residual vector field with explicit Euler
    steps. It returns a residual correction so it can plug into
    ProgramResidualWorldModel exactly like ResidualMLP.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (128, 128),
        activation: str = "silu",
        include_unknown_mask: bool = True,
        transition_dt: float = 0.05,
        ode_steps: int = 4,
        zero_init_output: bool = True,
    ) -> None:
        super().__init__()
        if state_dim <= 0 or action_dim <= 0:
            raise ValueError("state_dim and action_dim must be positive")
        if transition_dt <= 0:
            raise ValueError("transition_dt must be positive")
        if ode_steps <= 0:
            raise ValueError("ode_steps must be positive")
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.include_unknown_mask = bool(include_unknown_mask)
        self.transition_dt = float(transition_dt)
        self.ode_steps = int(ode_steps)
        self.residual_output_kind = "correction"

        input_dim = state_dim + action_dim + state_dim + 1
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
        current_states: torch.Tensor,
        actions: torch.Tensor,
        program_next_states: torch.Tensor,
        unknown_mask: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        features = [current_states, actions, program_next_states, tau]
        if self.include_unknown_mask:
            features.append(unknown_mask)
        return torch.cat(features, dim=-1)

    def residual_vector_field(
        self,
        current_states: torch.Tensor,
        actions: torch.Tensor,
        program_next_states: torch.Tensor,
        unknown_mask: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(
            self.build_features(
                current_states=current_states,
                actions=actions,
                program_next_states=program_next_states,
                unknown_mask=unknown_mask,
                tau=tau,
            )
        )

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        program_next_states: torch.Tensor,
        unknown_mask: torch.Tensor,
    ) -> torch.Tensor:
        symbolic_velocity = (program_next_states - states) / self.transition_dt
        current = states
        step_dt = self.transition_dt / self.ode_steps
        for step in range(self.ode_steps):
            tau_value = (step + 0.5) / self.ode_steps
            tau = torch.full(
                (states.shape[0], 1),
                tau_value,
                dtype=states.dtype,
                device=states.device,
            )
            residual_velocity = self.residual_vector_field(
                current_states=current,
                actions=actions,
                program_next_states=program_next_states,
                unknown_mask=unknown_mask,
                tau=tau,
            )
            current = current + step_dt * (symbolic_velocity + residual_velocity)
        return current - program_next_states


class DeltaGateMLP(nn.Module):
    """Dimension-wise reliability gate for symbolic delta predictions."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (128, 128),
        activation: str = "silu",
        initial_logit: float = -4.0,
    ) -> None:
        super().__init__()
        if state_dim <= 0 or action_dim <= 0:
            raise ValueError("state_dim and action_dim must be positive")
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)

        input_dim = state_dim + action_dim + state_dim + state_dim + state_dim
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(previous_dim, int(hidden_size)))
            layers.append(_activation_module(activation))
            previous_dim = int(hidden_size)
        output_layer = nn.Linear(previous_dim, state_dim)
        nn.init.zeros_(output_layer.weight)
        nn.init.constant_(output_layer.bias, float(initial_logit))
        layers.append(output_layer)
        self.net = nn.Sequential(*layers)

    def build_features(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        symbolic_delta: torch.Tensor,
        confidence: torch.Tensor,
        unknown_mask: torch.Tensor,
    ) -> torch.Tensor:
        known_mask = 1.0 - unknown_mask
        return torch.cat(
            [states, actions, symbolic_delta, confidence, known_mask],
            dim=-1,
        )

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        symbolic_delta: torch.Tensor,
        confidence: torch.Tensor,
        unknown_mask: torch.Tensor,
    ) -> torch.Tensor:
        features = self.build_features(
            states=states,
            actions=actions,
            symbolic_delta=symbolic_delta,
            confidence=confidence,
            unknown_mask=unknown_mask,
        )
        return torch.sigmoid(self.net(features))


class DiagonalVarianceMLP(nn.Module):
    """State-dependent diagonal log-variance head for probabilistic dynamics."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (128, 128),
        activation: str = "silu",
        min_log_variance: float = -8.0,
        max_log_variance: float = 4.0,
    ) -> None:
        super().__init__()
        if state_dim <= 0 or action_dim <= 0:
            raise ValueError("state_dim and action_dim must be positive")
        if min_log_variance >= max_log_variance:
            raise ValueError("min_log_variance must be smaller than max_log_variance")
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.min_log_variance = float(min_log_variance)
        self.max_log_variance = float(max_log_variance)

        input_dim = state_dim + action_dim + state_dim
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(previous_dim, int(hidden_size)))
            layers.append(_activation_module(activation))
            previous_dim = int(hidden_size)
        output_layer = nn.Linear(previous_dim, state_dim)
        nn.init.zeros_(output_layer.weight)
        nn.init.constant_(output_layer.bias, -4.0)
        layers.append(output_layer)
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        predictions: torch.Tensor,
    ) -> torch.Tensor:
        raw = self.net(torch.cat([states, actions, predictions], dim=-1))
        return raw.clamp(self.min_log_variance, self.max_log_variance)
