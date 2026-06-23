from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .law_dsl import LawPriorBank
from .templates import MechanismTemplate, prior_tensors


@dataclass(frozen=True)
class DUCWorldModelConfig:
    state_dim: int
    action_dim: int
    templates: tuple[MechanismTemplate, ...]
    hidden_size: int = 256
    hidden_layers: int = 2
    history_length: int = 4
    min_logvar: float = -8.0
    max_logvar: float = 2.0
    prior_beta_init: float = 1.0
    trust_region_delta_min: float = 0.15
    trust_region_delta_range: float = 0.75


@dataclass
class DUCForwardOutput:
    mean: torch.Tensor
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
    prior_delta: torch.Tensor
    residual_delta: torch.Tensor
    mechanism_delta: torch.Tensor
    prior_beta: torch.Tensor
    residual_scale: torch.Tensor
    prior_gate: torch.Tensor
    data_confidence: torch.Tensor


def _mlp(input_dim: int, output_dim: int, hidden_size: int, hidden_layers: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = input_dim
    for _ in range(hidden_layers):
        layers.append(nn.Linear(last, hidden_size))
        layers.append(nn.SiLU())
        last = hidden_size
    layers.append(nn.Linear(last, output_dim))
    return nn.Sequential(*layers)


class ResidualMechanismBank(nn.Module):
    """Neural residual mechanisms R_j that correct compiled prior laws P_j."""

    def __init__(self, config: DUCWorldModelConfig) -> None:
        super().__init__()
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.templates = config.templates
        for template in self.templates:
            template.validate(config.state_dim, config.action_dim)
        self.networks = nn.ModuleList()
        for template in self.templates:
            input_dim = len(template.state_indices) + len(template.action_indices)
            output_dim = len(template.output_indices)
            self.networks.append(
                _mlp(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    hidden_size=config.hidden_size,
                    hidden_layers=config.hidden_layers,
                )
            )

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        effects: list[torch.Tensor] = []
        for template, network in zip(self.templates, self.networks, strict=True):
            parts = []
            if template.state_indices:
                index = torch.tensor(template.state_indices, device=states.device)
                parts.append(states.index_select(dim=-1, index=index))
            if template.action_indices:
                index = torch.tensor(template.action_indices, device=actions.device)
                parts.append(actions.index_select(dim=-1, index=index))
            if parts:
                local_input = torch.cat(parts, dim=-1)
            else:
                local_input = states.new_zeros(states.shape[0], 1)
            local_effect = network(local_input)
            full_effect = local_effect.new_zeros(states.shape[0], self.state_dim)
            out_index = torch.tensor(template.output_indices, device=states.device)
            full_effect.index_copy_(dim=-1, index=out_index, source=local_effect)
            effects.append(full_effect)
        return torch.stack(effects, dim=1)


class ContextEncoder(nn.Module):
    def __init__(self, config: DUCWorldModelConfig) -> None:
        super().__init__()
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.history_length = config.history_length
        self.context_dim = len(config.templates)
        self.slow_indices = tuple(
            index
            for index, template in enumerate(config.templates)
            if template.timescale == "slow"
        )
        self.event_indices = tuple(
            index
            for index, template in enumerate(config.templates)
            if template.timescale in {"event", "unknown"}
        )
        slow_input_dim = 2 * (config.state_dim + config.action_dim)
        event_input_dim = config.history_length * (config.state_dim + config.action_dim)
        self.slow_network = (
            _mlp(
                input_dim=slow_input_dim,
                output_dim=2 * len(self.slow_indices),
                hidden_size=config.hidden_size,
                hidden_layers=max(1, config.hidden_layers),
            )
            if self.slow_indices
            else None
        )
        self.event_network = (
            _mlp(
                input_dim=event_input_dim,
                output_dim=2 * len(self.event_indices),
                hidden_size=config.hidden_size,
                hidden_layers=max(1, config.hidden_layers),
            )
            if self.event_indices
            else None
        )

    def forward(
        self,
        history_states: torch.Tensor,
        history_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if history_states.ndim != 3 or history_actions.ndim != 3:
            raise ValueError("history tensors must have shape [batch, history, dim]")
        batch_size = history_states.shape[0]
        mean = history_states.new_zeros(batch_size, self.context_dim)
        logvar = history_states.new_zeros(batch_size, self.context_dim)
        if self.slow_network is not None:
            slow_features = torch.cat(
                [
                    history_states.mean(dim=1),
                    history_actions.mean(dim=1),
                    history_states[:, -1] - history_states[:, 0],
                    history_actions[:, -1] - history_actions[:, 0],
                ],
                dim=-1,
            )
            slow_mean, slow_logvar = self.slow_network(slow_features).chunk(2, dim=-1)
            slow_index = torch.tensor(self.slow_indices, device=history_states.device)
            mean.index_copy_(dim=-1, index=slow_index, source=slow_mean.to(mean.dtype))
            logvar.index_copy_(dim=-1, index=slow_index, source=slow_logvar.to(logvar.dtype))
        if self.event_network is not None:
            event_features = torch.cat(
                [
                    history_states.reshape(batch_size, -1),
                    history_actions.reshape(batch_size, -1),
                ],
                dim=-1,
            )
            event_mean, event_logvar = self.event_network(event_features).chunk(2, dim=-1)
            event_index = torch.tensor(self.event_indices, device=history_states.device)
            mean.index_copy_(dim=-1, index=event_index, source=event_mean.to(mean.dtype))
            logvar.index_copy_(dim=-1, index=event_index, source=event_logvar.to(logvar.dtype))
        return mean, logvar.clamp(min=-8.0, max=4.0)


class DUCWorldModel(nn.Module):
    """Disentangled Universal Causal World Model.

    DUC-WM uses an LLM/template law DSL as a mechanistic prior P_j and lets a
    neural residual R_j repair what the prior misses:

        x_next = x + f_base(x, a)
                 + sum_j alpha_j(h) * (P_j(x, a, h) + R_j(x, a)).

    This keeps the prior executable, safe, and inspectable while preserving
    enough neural capacity for model mismatch.
    """

    def __init__(self, config: DUCWorldModelConfig) -> None:
        super().__init__()
        if not config.templates:
            raise ValueError("DUCWorldModel requires at least one mechanism template")
        self.config = config
        self.law_priors = LawPriorBank(config.templates, config.state_dim, config.action_dim)
        self.mechanisms = ResidualMechanismBank(config)
        self.context_encoder = ContextEncoder(config)
        self.base_dynamics = _mlp(
            input_dim=config.state_dim + config.action_dim,
            output_dim=config.state_dim,
            hidden_size=config.hidden_size,
            hidden_layers=config.hidden_layers,
        )
        self.variance_head = _mlp(
            input_dim=2 * config.state_dim + config.action_dim + len(config.templates),
            output_dim=config.state_dim,
            hidden_size=config.hidden_size,
            hidden_layers=max(1, config.hidden_layers - 1),
        )
        prior_mean, prior_std, scales, confidences = prior_tensors(config.templates)
        self.register_buffer("prior_mean", prior_mean)
        self.register_buffer("prior_std", prior_std)
        self.register_buffer("context_scales", scales)
        self.register_buffer("prior_confidence", confidences)
        self.register_buffer("prior_gate", torch.ones_like(confidences))
        self.register_buffer("data_confidence", torch.ones_like(confidences))
        beta_init = max(0.05, min(50.0, float(config.prior_beta_init)))
        self.prior_log_beta = nn.Parameter(
            torch.full(
                (len(config.templates),),
                float(torch.log(torch.tensor(beta_init))),
            )
        )
        self.register_buffer("_residual_scale", torch.tensor(1.0, dtype=torch.float32))
        self.unknown_indices = tuple(
            index
            for index, template in enumerate(config.templates)
            if template.timescale == "unknown" or template.name == "unknown"
        )

    @property
    def context_dim(self) -> int:
        return len(self.config.templates)

    @property
    def prior_beta(self) -> torch.Tensor:
        return torch.exp(self.prior_log_beta).clamp(min=0.05, max=50.0)

    @property
    def effective_prior_confidence(self) -> torch.Tensor:
        return (self.prior_confidence * self.data_confidence).clamp(min=0.0, max=1.0)

    @torch.no_grad()
    def set_prior_validation(
        self,
        gate: torch.Tensor,
        data_confidence: torch.Tensor,
        beta: torch.Tensor | None = None,
    ) -> None:
        gate = gate.to(device=self.prior_gate.device, dtype=self.prior_gate.dtype)
        data_confidence = data_confidence.to(
            device=self.data_confidence.device,
            dtype=self.data_confidence.dtype,
        )
        if gate.shape != self.prior_gate.shape:
            raise ValueError(f"prior gate has shape {tuple(gate.shape)}, expected {tuple(self.prior_gate.shape)}")
        if data_confidence.shape != self.data_confidence.shape:
            raise ValueError(
                "prior data confidence has shape "
                f"{tuple(data_confidence.shape)}, expected {tuple(self.data_confidence.shape)}"
            )
        self.prior_gate.copy_(gate.clamp(min=0.0, max=1.0))
        self.data_confidence.copy_(data_confidence.clamp(min=0.0, max=1.0))
        if beta is not None:
            beta = beta.to(device=self.prior_log_beta.device, dtype=self.prior_log_beta.dtype)
            if beta.shape != self.prior_log_beta.shape:
                raise ValueError(f"prior beta has shape {tuple(beta.shape)}, expected {tuple(self.prior_log_beta.shape)}")
            beta = beta.clamp(min=0.05, max=50.0)
            self.prior_log_beta.copy_(torch.log(beta))

    def set_residual_scale(self, value: float) -> None:
        value = float(max(0.0, min(1.0, value)))
        self._residual_scale.fill_(value)

    def alpha_from_raw(self, raw_context: torch.Tensor) -> torch.Tensor:
        return self.context_scales.to(raw_context.device) * torch.tanh(raw_context)

    def default_history(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        history_states = states.unsqueeze(1).expand(-1, self.config.history_length, -1)
        history_actions = actions.unsqueeze(1).expand(-1, self.config.history_length, -1)
        return history_states, history_actions

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        history_states: torch.Tensor | None = None,
        history_actions: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        sample_context: bool = True,
    ) -> DUCForwardOutput:
        if history_states is None or history_actions is None:
            history_states, history_actions = self.default_history(states, actions)
        posterior_mean, posterior_logvar = self.context_encoder(history_states, history_actions)
        alpha_mean = self.alpha_from_raw(posterior_mean)
        if context is not None:
            alpha = context
        elif sample_context and self.training:
            std = torch.exp(0.5 * posterior_logvar)
            raw = posterior_mean + std * torch.randn_like(std)
            alpha = self.alpha_from_raw(raw)
        else:
            alpha = alpha_mean

        base_input = torch.cat([states, actions], dim=-1)
        base_delta = self.base_dynamics(base_input).to(states.dtype)
        raw_prior_effects = self.law_priors(
            states,
            actions,
            history_states=history_states,
            history_actions=history_actions,
        )
        beta = self.prior_beta.to(raw_prior_effects.device, raw_prior_effects.dtype)
        gate = self.prior_gate.to(raw_prior_effects.device, raw_prior_effects.dtype)
        prior_effects = raw_prior_effects * beta.view(1, -1, 1) * gate.view(1, -1, 1)
        raw_residual_effects = self.mechanisms(states, actions)
        residual_scale = self._residual_scale.to(raw_residual_effects.device, raw_residual_effects.dtype)
        residual_effects = residual_scale * raw_residual_effects
        effects = prior_effects + residual_effects
        prior_delta = torch.einsum("bk,bkd->bd", alpha, prior_effects).to(states.dtype)
        residual_delta = torch.einsum("bk,bkd->bd", alpha, residual_effects).to(states.dtype)
        mechanism_delta = prior_delta + residual_delta
        mean = states + base_delta + mechanism_delta
        logvar_input = torch.cat([states, actions, alpha, base_delta], dim=-1)
        logvar = self.variance_head(logvar_input).clamp(
            min=self.config.min_logvar,
            max=self.config.max_logvar,
        )
        return DUCForwardOutput(
            mean=mean,
            logvar=logvar,
            effects=effects,
            prior_effects=prior_effects,
            residual_effects=residual_effects,
            raw_prior_effects=raw_prior_effects,
            raw_residual_effects=raw_residual_effects,
            alpha=alpha,
            alpha_mean=alpha_mean,
            posterior_mean=posterior_mean,
            posterior_logvar=posterior_logvar,
            base_delta=base_delta,
            prior_delta=prior_delta,
            residual_delta=residual_delta,
            mechanism_delta=mechanism_delta,
            prior_beta=beta,
            residual_scale=residual_scale,
            prior_gate=gate,
            data_confidence=self.data_confidence.to(raw_prior_effects.device, raw_prior_effects.dtype),
        )

    def nll(self, output: DUCForwardOutput, targets: torch.Tensor) -> torch.Tensor:
        inv_var = torch.exp(-output.logvar)
        return 0.5 * ((targets - output.mean).pow(2) * inv_var + output.logvar).mean()
