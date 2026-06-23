from __future__ import annotations

import torch
import torch.nn as nn

from .templates import MechanismTemplate


class LawPriorBank(nn.Module):
    """Compile safe LLM/template law-DSL entries into tensor prior effects.

    The LLM never emits executable Python. It emits a bounded `law_type` plus
    masks and gains. This module turns that declarative prior into P_j(x,a,h),
    which is then corrected by a neural residual mechanism.
    """

    def __init__(
        self,
        templates: tuple[MechanismTemplate, ...],
        state_dim: int,
        action_dim: int,
    ) -> None:
        super().__init__()
        self.templates = templates
        self.state_dim = state_dim
        self.action_dim = action_dim
        for template in templates:
            template.validate(state_dim, action_dim)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        history_states: torch.Tensor | None = None,
        history_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        effects = [
            self._template_effect(template, states, actions, history_states, history_actions)
            for template in self.templates
        ]
        return torch.stack(effects, dim=1)

    def _template_effect(
        self,
        template: MechanismTemplate,
        states: torch.Tensor,
        actions: torch.Tensor,
        history_states: torch.Tensor | None,
        history_actions: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size = int(states.shape[0])
        out_index = _index(template.output_indices, states.device)
        local_dim = len(template.output_indices)
        law_type = template.law_type
        if law_type == "learned_residual":
            local = states.new_zeros(batch_size, local_dim)
        elif law_type == "actuation":
            local = _project_actions(actions, local_dim)
        elif law_type == "external_drift":
            local = states.new_ones(batch_size, local_dim)
        elif law_type == "velocity_damping":
            local = -states.index_select(dim=-1, index=out_index)
        elif law_type == "inertia_shift":
            local = _project_actions(actions, local_dim) * _state_energy(states)
        elif law_type == "action_delay":
            previous = _previous_action(actions, history_actions)
            local = _project_actions(previous - actions, local_dim)
        elif law_type == "sticky_velocity":
            previous_state = _previous_state(states, history_states)
            local = -(states - previous_state).index_select(dim=-1, index=out_index)
        elif law_type == "impulse":
            local = torch.tanh(states.index_select(dim=-1, index=out_index))
        elif law_type == "gravity_shift":
            local = -states.new_ones(batch_size, local_dim)
        else:
            raise ValueError(f"unsupported law_type={law_type!r}")

        local = local * float(template.law_gain)
        full = local.new_zeros(batch_size, self.state_dim)
        full.index_copy_(dim=-1, index=out_index, source=local.to(full.dtype))
        return full


def _index(indices: tuple[int, ...], device: torch.device) -> torch.Tensor:
    return torch.tensor(indices, dtype=torch.long, device=device)


def _project_actions(actions: torch.Tensor, output_dim: int) -> torch.Tensor:
    if actions.shape[-1] == 0:
        return actions.new_zeros(actions.shape[0], output_dim)
    value = torch.tanh(actions)
    if value.shape[-1] == output_dim:
        return value
    repeat = (output_dim + value.shape[-1] - 1) // value.shape[-1]
    return value.repeat(1, repeat)[:, :output_dim]


def _state_energy(states: torch.Tensor) -> torch.Tensor:
    return 1.0 + 0.25 * torch.tanh(states.pow(2).mean(dim=-1, keepdim=True))


def _previous_action(
    actions: torch.Tensor,
    history_actions: torch.Tensor | None,
) -> torch.Tensor:
    if history_actions is None or history_actions.shape[1] < 2:
        return actions.new_zeros(actions.shape)
    return history_actions[:, -2].to(actions.dtype)


def _previous_state(
    states: torch.Tensor,
    history_states: torch.Tensor | None,
) -> torch.Tensor:
    if history_states is None or history_states.shape[1] < 2:
        return states
    return history_states[:, -2].to(states.dtype)
