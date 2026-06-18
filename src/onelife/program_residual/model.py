from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

from .core import ModelOutput
from .program import SymbolicProgram


class ProgramResidualWorldModel(nn.Module):
    """Continuous world model: executable program plus neural residual correction."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        program: SymbolicProgram,
        residual_model: nn.Module,
        gate_model: nn.Module | None = None,
        variance_model: nn.Module | None = None,
        apply_unknown_mask: bool = False,
    ) -> None:
        super().__init__()
        if state_dim <= 0 or action_dim <= 0:
            raise ValueError("state_dim and action_dim must be positive")
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.program = program
        self.residual_model = residual_model
        self.gate_model = gate_model
        self.variance_model = variance_model
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
        symbolic_gate = None
        if self.gate_model is None:
            applied_residual = (
                program_output.unknown_mask * residual
                if self.apply_unknown_mask
                else residual
            )
            prediction = program_output.next_state + applied_residual
        else:
            symbolic_delta = program_output.next_state - states_batched
            symbolic_candidate_delta = symbolic_delta + residual

            # Compute the same neural backbone with a neutral symbolic program.
            # This makes gate=0 an exact fallback to the neural-only dynamics.
            neutral_program_next = states_batched
            neutral_unknown_mask = torch.ones_like(program_output.unknown_mask)
            neural_residual = self.residual_model(
                states_batched,
                actions_batched,
                neutral_program_next,
                neutral_unknown_mask,
            )
            neural_delta = neural_residual

            raw_gate = self.gate_model(
                states_batched,
                actions_batched,
                symbolic_delta,
                program_output.confidence,
                program_output.unknown_mask,
            )
            graph_budget = (
                program_output.graph_budget
                if program_output.graph_budget is not None
                else 1.0 - program_output.unknown_mask
            )
            symbolic_gate = raw_gate * graph_budget * (1.0 - program_output.unknown_mask)
            intervention_delta = symbolic_gate * (
                symbolic_candidate_delta - neural_delta
            )
            applied_residual = neural_delta + intervention_delta
            prediction = states_batched + applied_residual
            residual = neural_delta

        if squeeze:
            prediction = prediction.squeeze(0)
            program_next_state = program_output.next_state.squeeze(0)
            residual = residual.squeeze(0)
            applied_residual = applied_residual.squeeze(0)
            confidence = program_output.confidence.squeeze(0)
            unknown_mask = program_output.unknown_mask.squeeze(0)
            program_variance = (
                program_output.variance.squeeze(0)
                if program_output.variance is not None
                else None
            )
            if symbolic_gate is not None:
                symbolic_gate = symbolic_gate.squeeze(0)
            graph_budget = (
                program_output.graph_budget.squeeze(0)
                if program_output.graph_budget is not None
                else None
            )
        else:
            program_next_state = program_output.next_state
            confidence = program_output.confidence
            unknown_mask = program_output.unknown_mask
            program_variance = program_output.variance
            graph_budget = program_output.graph_budget

        log_variance = None
        if self.variance_model is not None:
            log_variance = self.variance_model(
                states_batched,
                actions_batched,
                prediction if not squeeze else prediction.unsqueeze(0),
            )
            if squeeze:
                log_variance = log_variance.squeeze(0)

        return ModelOutput(
            prediction=prediction,
            program_next_state=program_next_state,
            residual=residual,
            applied_residual=applied_residual,
            confidence=confidence,
            unknown_mask=unknown_mask,
            active_laws=program_output.active_laws,
            symbolic_gate=symbolic_gate,
            graph_budget=graph_budget,
            program_variance=program_variance,
            log_variance=log_variance,
        )

    def symbolic_weight_l1(self) -> torch.Tensor:
        return self.program.symbolic_weight_l1()

    @torch.no_grad()
    def predict_next_state(
        self,
        state: torch.Tensor | npt.NDArray[np.float32],
        action: torch.Tensor | npt.NDArray[np.float32],
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        input_device = state.device if isinstance(state, torch.Tensor) else torch.device("cpu")
        model_device = _module_device(self)
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=model_device)
        action_tensor = torch.as_tensor(action, dtype=torch.float32, device=model_device)
        output = self(state_tensor, action_tensor)
        if was_training:
            self.train()
        return output.prediction.to(input_device)


def _module_device(module: nn.Module) -> torch.device:
    parameter = next(module.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(module.buffers(), None)
    if buffer is not None:
        return buffer.device
    return torch.device("cpu")
