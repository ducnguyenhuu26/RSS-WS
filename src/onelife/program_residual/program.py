from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from .core import ContinuousLawProtocol, ProgramOutput


class SymbolicProgram(nn.Module):
    """
    Executable continuous symbolic dynamics program.

    The program applies active laws to a single state-action pair, combines
    overlapping predictions by confidence-weighted averaging, and marks
    dimensions with no confident symbolic prediction as unknown for the neural
    residual.
    """

    def __init__(
        self,
        state_dim: int,
        laws: Sequence[ContinuousLawProtocol] | None = None,
        unknown_confidence_threshold: float = 1e-6,
        identity_for_unknown: bool = True,
    ) -> None:
        super().__init__()
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")
        self.state_dim = int(state_dim)
        self.unknown_confidence_threshold = float(unknown_confidence_threshold)
        self.identity_for_unknown = bool(identity_for_unknown)

        module_laws: list[nn.Module] = []
        for law in laws or []:
            if not isinstance(law, nn.Module):
                raise TypeError(
                    "SymbolicProgram laws must inherit torch.nn.Module "
                    "so trainable symbolic parameters are registered."
                )
            module_laws.append(law)
        self.laws = nn.ModuleList(module_laws)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> ProgramOutput:
        return self.predict(states, actions)

    def predict(self, states: torch.Tensor, actions: torch.Tensor) -> ProgramOutput:
        squeeze = states.ndim == 1
        states_batched = states.unsqueeze(0) if squeeze else states
        actions_batched = actions.unsqueeze(0) if actions.ndim == 1 else actions

        if states_batched.ndim != 2:
            raise ValueError("states must have shape [state_dim] or [batch, state_dim]")
        if actions_batched.ndim != 2:
            raise ValueError(
                "actions must have shape [action_dim] or [batch, action_dim]"
            )
        if states_batched.shape[0] != actions_batched.shape[0]:
            raise ValueError("states and actions must have the same batch size")
        if states_batched.shape[1] != self.state_dim:
            raise ValueError(
                f"expected state_dim={self.state_dim}, got {states_batched.shape[1]}"
            )

        outputs = [
            self._predict_one(state, action)
            for state, action in zip(states_batched, actions_batched)
        ]
        next_state = torch.stack([output.next_state for output in outputs], dim=0)
        confidence = torch.stack([output.confidence for output in outputs], dim=0)
        unknown_mask = torch.stack([output.unknown_mask for output in outputs], dim=0)
        active_laws = tuple(output.active_laws[0] for output in outputs)

        if squeeze:
            next_state = next_state.squeeze(0)
            confidence = confidence.squeeze(0)
            unknown_mask = unknown_mask.squeeze(0)

        return ProgramOutput(
            next_state=next_state,
            confidence=confidence,
            unknown_mask=unknown_mask,
            active_laws=active_laws,
        )

    def _predict_one(self, state: torch.Tensor, action: torch.Tensor) -> ProgramOutput:
        numerator = torch.zeros(self.state_dim, dtype=state.dtype, device=state.device)
        denominator = torch.zeros_like(numerator)
        active_laws: list[str] = []

        for law in self.laws:
            if not law.precondition(state, action):  # type: ignore[attr-defined]
                continue
            prediction = law.predict(state, action)  # type: ignore[attr-defined]
            indices = prediction.indices.to(device=state.device, dtype=torch.long)
            values = prediction.values.to(device=state.device, dtype=state.dtype)
            confidence = prediction.confidence.to(
                device=state.device,
                dtype=state.dtype,
            )
            if indices.ndim != 1:
                raise ValueError("law prediction indices must be one-dimensional")
            if values.shape != indices.shape or confidence.shape != indices.shape:
                raise ValueError("law prediction values/confidence must match indices")
            if torch.any(indices < 0) or torch.any(indices >= self.state_dim):
                raise ValueError("law prediction index out of bounds")

            numerator.index_add_(0, indices, values * confidence)
            denominator.index_add_(0, indices, confidence)
            active_laws.append(prediction.law_name)

        known = denominator > self.unknown_confidence_threshold
        if self.identity_for_unknown:
            next_state = state.clone()
        else:
            next_state = torch.zeros_like(state)
        next_state = torch.where(
            known,
            numerator / denominator.clamp_min(1e-12),
            next_state,
        )
        unknown_mask = (~known).to(dtype=state.dtype)

        return ProgramOutput(
            next_state=next_state,
            confidence=denominator,
            unknown_mask=unknown_mask,
            active_laws=(tuple(active_laws),),
        )
