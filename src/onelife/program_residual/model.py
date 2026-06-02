from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

from .core import ModelOutput
from .program import SymbolicProgram


class ProgramResidualWorldModel(nn.Module):
    """Continuous world model: executable program plus masked neural residual."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        program: SymbolicProgram,
        residual_model: nn.Module,
        apply_unknown_mask: bool = True,
    ) -> None:
        super().__init__()
        if state_dim <= 0 or action_dim <= 0:
            raise ValueError("state_dim and action_dim must be positive")
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.program = program
        self.residual_model = residual_model
        self.apply_unknown_mask = bool(apply_unknown_mask)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> ModelOutput:
        squeeze = states.ndim == 1
        states_batched = states.unsqueeze(0) if squeeze else states
        actions_batched = actions.unsqueeze(0) if actions.ndim == 1 else actions

        if states_batched.ndim != 2:
            raise ValueError("states must have shape [state_dim] or [batch, state_dim]")
        if actions_batched.ndim != 2:
            raise ValueError(
                "actions must have shape [action_dim] or [batch, action_dim]"
            )
        if states_batched.shape[1] != self.state_dim:
            raise ValueError(
                f"expected state_dim={self.state_dim}, got {states_batched.shape[1]}"
            )
        if actions_batched.shape[1] != self.action_dim:
            raise ValueError(
                f"expected action_dim={self.action_dim}, got {actions_batched.shape[1]}"
            )

        program_output = self.program(states_batched, actions_batched)
        residual = self.residual_model(
            states_batched,
            actions_batched,
            program_output.next_state,
            program_output.unknown_mask,
        )
        applied_residual = (
            program_output.unknown_mask * residual
            if self.apply_unknown_mask
            else residual
        )
        prediction = program_output.next_state + applied_residual

        if squeeze:
            prediction = prediction.squeeze(0)
            program_next_state = program_output.next_state.squeeze(0)
            residual = residual.squeeze(0)
            applied_residual = applied_residual.squeeze(0)
            confidence = program_output.confidence.squeeze(0)
            unknown_mask = program_output.unknown_mask.squeeze(0)
        else:
            program_next_state = program_output.next_state
            confidence = program_output.confidence
            unknown_mask = program_output.unknown_mask

        return ModelOutput(
            prediction=prediction,
            program_next_state=program_next_state,
            residual=residual,
            applied_residual=applied_residual,
            confidence=confidence,
            unknown_mask=unknown_mask,
            active_laws=program_output.active_laws,
        )

    @torch.no_grad()
    def predict_next_state(
        self,
        state: torch.Tensor | npt.NDArray[np.float32],
        action: torch.Tensor | npt.NDArray[np.float32],
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        state_tensor = torch.as_tensor(state, dtype=torch.float32)
        action_tensor = torch.as_tensor(action, dtype=torch.float32)
        output = self(state_tensor, action_tensor)
        if was_training:
            self.train()
        return output.prediction
