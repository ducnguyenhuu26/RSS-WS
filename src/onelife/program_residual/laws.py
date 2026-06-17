from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from .core import LawPrediction


def _index_tensor(indices: Sequence[int]) -> torch.Tensor:
    return torch.tensor(list(indices), dtype=torch.long)


class ContinuousLaw(nn.Module):
    """Base class for continuous program laws."""

    @property
    def law_name(self) -> str:
        return self.__class__.__name__

    def precondition(self, state: torch.Tensor, action: torch.Tensor) -> bool:
        return True

    def predict(self, state: torch.Tensor, action: torch.Tensor) -> LawPrediction:
        raise NotImplementedError


class KinematicPositionLaw(ContinuousLaw):
    """Euler position update: q_next = q + dt * qdot."""

    def __init__(
        self,
        position_indices: Sequence[int],
        velocity_indices: Sequence[int],
        dt: float,
        confidence: float = 1.0,
        name: str | None = None,
    ) -> None:
        super().__init__()
        if len(position_indices) != len(velocity_indices):
            raise ValueError("position_indices and velocity_indices must match length")
        self.register_buffer("position_indices", _index_tensor(position_indices))
        self.register_buffer("velocity_indices", _index_tensor(velocity_indices))
        self.dt = float(dt)
        self.confidence_value = float(confidence)
        self._name = name

    @property
    def law_name(self) -> str:
        return self._name or super().law_name

    def predict(self, state: torch.Tensor, action: torch.Tensor) -> LawPrediction:
        indices = self.position_indices.to(device=state.device)
        velocity_indices = self.velocity_indices.to(device=state.device)
        values = state[indices] + self.dt * state[velocity_indices]
        confidence = torch.full_like(values, self.confidence_value)
        return LawPrediction(
            indices=indices,
            values=values,
            confidence=confidence,
            law_name=self.law_name,
            value_kind="next_state",
        )


class LinearVelocityLaw(ContinuousLaw):
    """Trainable linear velocity update with action gain and damping."""

    def __init__(
        self,
        velocity_indices: Sequence[int],
        action_dim: int,
        dt: float,
        confidence: float = 1.0,
        trainable: bool = True,
        name: str | None = None,
    ) -> None:
        super().__init__()
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        velocity_indices_tensor = _index_tensor(velocity_indices)
        if velocity_indices_tensor.numel() == 0:
            raise ValueError("velocity_indices must not be empty")

        self.register_buffer("velocity_indices", velocity_indices_tensor)
        self.dt = float(dt)
        self.confidence_value = float(confidence)
        self._name = name

        num_velocities = int(velocity_indices_tensor.numel())
        action_gain = torch.zeros(num_velocities, action_dim, dtype=torch.float32)
        damping = torch.zeros(num_velocities, dtype=torch.float32)
        if trainable:
            self.action_gain = nn.Parameter(action_gain)
            self.damping = nn.Parameter(damping)
        else:
            self.register_buffer("action_gain", action_gain)
            self.register_buffer("damping", damping)

    @property
    def law_name(self) -> str:
        return self._name or super().law_name

    def predict(self, state: torch.Tensor, action: torch.Tensor) -> LawPrediction:
        indices = self.velocity_indices.to(device=state.device)
        current_velocity = state[indices]
        action_effect = self.action_gain.to(device=state.device) @ action
        damping_effect = self.damping.to(device=state.device) * current_velocity
        values = current_velocity + self.dt * (action_effect - damping_effect)
        confidence = torch.full_like(values, self.confidence_value)
        return LawPrediction(
            indices=indices,
            values=values,
            confidence=confidence,
            law_name=self.law_name,
            value_kind="next_state",
        )


class JointLimitVelocityLaw(ContinuousLaw):
    """Clamp selected velocities when corresponding positions exceed limits."""

    def __init__(
        self,
        position_indices: Sequence[int],
        velocity_indices: Sequence[int],
        lower_limits: Sequence[float],
        upper_limits: Sequence[float],
        confidence: float = 1.0,
        name: str | None = None,
    ) -> None:
        super().__init__()
        if not (
            len(position_indices)
            == len(velocity_indices)
            == len(lower_limits)
            == len(upper_limits)
        ):
            raise ValueError("all joint limit inputs must match length")
        self.register_buffer("position_indices", _index_tensor(position_indices))
        self.register_buffer("velocity_indices", _index_tensor(velocity_indices))
        self.register_buffer(
            "lower_limits", torch.tensor(list(lower_limits), dtype=torch.float32)
        )
        self.register_buffer(
            "upper_limits", torch.tensor(list(upper_limits), dtype=torch.float32)
        )
        self.confidence_value = float(confidence)
        self._name = name

    @property
    def law_name(self) -> str:
        return self._name or super().law_name

    def precondition(self, state: torch.Tensor, action: torch.Tensor) -> bool:
        positions = state[self.position_indices.to(device=state.device)]
        lower_limits = self.lower_limits.to(device=state.device)
        upper_limits = self.upper_limits.to(device=state.device)
        return bool(torch.any((positions <= lower_limits) | (positions >= upper_limits)))

    def predict(self, state: torch.Tensor, action: torch.Tensor) -> LawPrediction:
        position_indices = self.position_indices.to(device=state.device)
        velocity_indices = self.velocity_indices.to(device=state.device)
        lower_limits = self.lower_limits.to(device=state.device)
        upper_limits = self.upper_limits.to(device=state.device)

        positions = state[position_indices]
        current_velocity = state[velocity_indices]
        at_lower = positions <= lower_limits
        at_upper = positions >= upper_limits
        blocked_lower_velocity = torch.clamp(current_velocity, min=0.0)
        blocked_upper_velocity = torch.clamp(current_velocity, max=0.0)
        values = torch.where(at_lower, blocked_lower_velocity, current_velocity)
        values = torch.where(at_upper, blocked_upper_velocity, values)
        confidence = torch.full_like(values, self.confidence_value)
        return LawPrediction(
            indices=velocity_indices,
            values=values,
            confidence=confidence,
            law_name=self.law_name,
            value_kind="next_state",
        )
