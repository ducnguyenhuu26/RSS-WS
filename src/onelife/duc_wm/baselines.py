from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from onelife.mujoco_dataset import MuJoCoTransitions

from .core import mlp, weighted_mse
from .data import (
    DUCBatch,
    align_contexts_to_templates,
    iter_duc_batches,
    iter_prepared_duc_batches,
    prepare_duc_data,
)
from .law_dsl import LawPriorBank
from .metrics import _history_for_indices, default_control_weights, evaluate_world_model
from .templates import MechanismTemplate, prior_tensors


@dataclass(frozen=True)
class BaselineTrainerConfig:
    epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 1e-3
    history_length: int = 4
    control_weight: float = 0.0
    rollout_weight: float = 0.0
    rollout_horizon: int = 1
    seed: int = 0
    precision: str = "fp32"
    preload_to_device: bool = False


@dataclass(frozen=True)
class PETSWorldModelConfig:
    state_dim: int
    action_dim: int
    context_dim: int = 0
    hidden_size: int = 256
    hidden_layers: int = 2
    ensemble_size: int = 5
    min_logvar: float = -8.0
    max_logvar: float = 2.0


@dataclass(frozen=True)
class MLPWorldModelConfig:
    state_dim: int
    action_dim: int
    context_dim: int = 0
    hidden_size: int = 256
    hidden_layers: int = 2
    min_logvar: float = -8.0
    max_logvar: float = 2.0


@dataclass(frozen=True)
class CaDMWorldModelConfig:
    state_dim: int
    action_dim: int
    history_length: int = 4
    context_dim: int = 16
    hidden_size: int = 256
    hidden_layers: int = 2
    min_logvar: float = -8.0
    max_logvar: float = 2.0


@dataclass(frozen=True)
class LEANWorldModelConfig:
    state_dim: int
    action_dim: int
    templates: tuple[MechanismTemplate, ...]
    mode: str = "gr"
    history_length: int = 4
    hidden_size: int = 256
    hidden_layers: int = 2
    min_logvar: float = -8.0
    max_logvar: float = 2.0
    symbolic_delta_scale: float = 0.20
    symbolic_var_floor: float = 0.02
    symbolic_var_ceiling: float = 25.0
    symbolic_temperature: float = 2.0
    trust_scale: float = 1.0


@dataclass
class BaselineForwardOutput:
    mean: torch.Tensor
    logvar: torch.Tensor
    latent: torch.Tensor | None = None
    member_means: torch.Tensor | None = None
    member_logvars: torch.Tensor | None = None


class MLPWorldModel(nn.Module):
    """Single black-box Gaussian delta model.

    This is the simplest capacity-controlled dynamics baseline: no ensemble,
    no context encoder, no named mechanisms, no LLM prior.
    """

    def __init__(self, config: MLPWorldModelConfig) -> None:
        super().__init__()
        self.config = config
        self.network = mlp(
            input_dim=config.state_dim + config.action_dim + config.context_dim,
            output_dim=2 * config.state_dim,
            hidden_size=config.hidden_size,
            hidden_layers=config.hidden_layers,
        )

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        history_states: torch.Tensor | None = None,
        history_actions: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        sample_context: bool = False,
    ) -> BaselineForwardOutput:
        del history_states, history_actions, sample_context
        inputs = _context_inputs(states, actions, context, self.config.context_dim)
        delta, logvar = self.network(inputs).chunk(2, dim=-1)
        return BaselineForwardOutput(
            mean=states + delta,
            logvar=logvar.clamp(self.config.min_logvar, self.config.max_logvar),
        )

    def nll(self, output: BaselineForwardOutput, targets: torch.Tensor) -> torch.Tensor:
        inv_var = torch.exp(-output.logvar)
        return 0.5 * ((targets - output.mean).pow(2) * inv_var + output.logvar).mean()


class PETSWorldModel(nn.Module):
    """PETS-style probabilistic ensemble dynamics model.

    This is the workshop baseline version: an ensemble of Gaussian delta models
    trained on the same offline transitions and evaluated with the same rollout
    metric as SimFutures-LP. It does not implement particle TS inside MPC.
    """

    def __init__(self, config: PETSWorldModelConfig) -> None:
        super().__init__()
        if config.ensemble_size <= 0:
            raise ValueError("ensemble_size must be positive")
        self.config = config
        output_dim = 2 * config.state_dim
        self.members = nn.ModuleList(
            [
                mlp(
                    input_dim=config.state_dim + config.action_dim + config.context_dim,
                    output_dim=output_dim,
                    hidden_size=config.hidden_size,
                    hidden_layers=config.hidden_layers,
                )
                for _ in range(config.ensemble_size)
            ]
        )

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        history_states: torch.Tensor | None = None,
        history_actions: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        sample_context: bool = False,
    ) -> BaselineForwardOutput:
        del history_states, history_actions, sample_context
        inputs = _context_inputs(states, actions, context, self.config.context_dim)
        means: list[torch.Tensor] = []
        logvars: list[torch.Tensor] = []
        for member in self.members:
            delta, logvar = member(inputs).chunk(2, dim=-1)
            means.append(states + delta)
            logvars.append(logvar.clamp(self.config.min_logvar, self.config.max_logvar))
        member_means = torch.stack(means, dim=0)
        member_logvars = torch.stack(logvars, dim=0)
        mean = member_means.mean(dim=0)
        # Moment-matched predictive variance: aleatoric + epistemic.
        member_vars = torch.exp(member_logvars)
        variance = (member_vars + member_means.pow(2)).mean(dim=0) - mean.pow(2)
        logvar = variance.clamp_min(1e-8).log().clamp(
            self.config.min_logvar,
            self.config.max_logvar,
        )
        return BaselineForwardOutput(
            mean=mean,
            logvar=logvar,
            member_means=member_means,
            member_logvars=member_logvars,
        )

    def nll(self, output: BaselineForwardOutput, targets: torch.Tensor) -> torch.Tensor:
        if output.member_means is None or output.member_logvars is None:
            raise ValueError("PETS nll requires member predictions")
        inv_var = torch.exp(-output.member_logvars)
        nll = 0.5 * (
            (targets.unsqueeze(0) - output.member_means).pow(2) * inv_var
            + output.member_logvars
        )
        return nll.mean()


class CaDMWorldModel(nn.Module):
    """CaDM-style latent-context dynamics model.

    It learns an uninterpreted context vector from recent history and conditions
    a Gaussian delta model on that vector. There are no named mechanisms or LLM
    masks, which makes it the closest architecture-level baseline for SimFutures-LP.
    """

    def __init__(self, config: CaDMWorldModelConfig) -> None:
        super().__init__()
        self.config = config
        history_input_dim = config.history_length * (config.state_dim + config.action_dim)
        self.context_encoder = mlp(
            input_dim=history_input_dim,
            output_dim=config.context_dim,
            hidden_size=config.hidden_size,
            hidden_layers=max(1, config.hidden_layers),
        )
        self.dynamics = mlp(
            input_dim=config.state_dim + config.action_dim + config.context_dim,
            output_dim=2 * config.state_dim,
            hidden_size=config.hidden_size,
            hidden_layers=config.hidden_layers,
        )

    def default_history(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        history_states = states.unsqueeze(1).expand(-1, self.config.history_length, -1)
        history_actions = actions.unsqueeze(1).expand(-1, self.config.history_length, -1)
        return history_states, history_actions

    def encode_context(
        self,
        history_states: torch.Tensor,
        history_actions: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat(
            [
                history_states.reshape(history_states.shape[0], -1),
                history_actions.reshape(history_actions.shape[0], -1),
            ],
            dim=-1,
        )
        return self.context_encoder(features)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        history_states: torch.Tensor | None = None,
        history_actions: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        sample_context: bool = False,
    ) -> BaselineForwardOutput:
        del sample_context
        if context is None:
            if history_states is None or history_actions is None:
                history_states, history_actions = self.default_history(states, actions)
            context = self.encode_context(history_states, history_actions)
        inputs = torch.cat([states, actions, context], dim=-1)
        delta, logvar = self.dynamics(inputs).chunk(2, dim=-1)
        return BaselineForwardOutput(
            mean=states + delta,
            logvar=logvar.clamp(self.config.min_logvar, self.config.max_logvar),
            latent=context,
        )

    def nll(self, output: BaselineForwardOutput, targets: torch.Tensor) -> torch.Tensor:
        inv_var = torch.exp(-output.logvar)
        return 0.5 * ((targets - output.mean).pow(2) * inv_var + output.logvar).mean()


class LEANWorldModel(nn.Module):
    """Neurosymbolic gated-residual/trust-region baseline.

    The model receives the same executable law portfolio as SimFutures-LP, but
    it does not use wake replay, law-posterior calibration, planning bonus, or
    certified-risk calibration. A history-conditioned neural backbone predicts
    the transition; symbolic laws add only a gated correction.
    """

    def __init__(self, config: LEANWorldModelConfig) -> None:
        super().__init__()
        if not config.templates:
            raise ValueError("LEANWorldModel requires at least one law template")
        if config.mode not in {"gr", "tr"}:
            raise ValueError("LEANWorldModel mode must be 'gr' or 'tr'")
        self.config = config
        self.law_priors = LawPriorBank(config.templates, config.state_dim, config.action_dim)
        self.num_laws = len(config.templates)
        history_dim = config.history_length * (config.state_dim + config.action_dim)
        base_input_dim = history_dim + config.state_dim + config.action_dim
        conditioned_dim = base_input_dim + self.num_laws + 2 * config.state_dim
        self.backbone = mlp(
            input_dim=base_input_dim,
            output_dim=2 * config.state_dim,
            hidden_size=config.hidden_size,
            hidden_layers=config.hidden_layers,
        )
        self.law_activation = mlp(
            input_dim=base_input_dim,
            output_dim=self.num_laws,
            hidden_size=config.hidden_size,
            hidden_layers=max(1, config.hidden_layers),
        )
        self.gate = mlp(
            input_dim=conditioned_dim,
            output_dim=config.state_dim,
            hidden_size=config.hidden_size,
            hidden_layers=max(1, config.hidden_layers),
        )
        self.trust_correction = mlp(
            input_dim=conditioned_dim,
            output_dim=config.state_dim,
            hidden_size=config.hidden_size,
            hidden_layers=max(1, config.hidden_layers),
        )
        prior_mean, prior_std, scales, confidences = prior_tensors(config.templates)
        self.register_buffer("law_prior_mean", prior_mean)
        self.register_buffer("law_prior_std", prior_std)
        self.register_buffer("context_scales", scales)
        self.register_buffer("law_confidence", confidences)
        self.register_buffer("law_logit_bias", torch.logit(confidences.clamp(1e-6, 1.0 - 1e-6)))
        self.register_buffer("law_output_mask", _law_output_mask(config.templates, config.state_dim))

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
        sample_context: bool = False,
    ) -> BaselineForwardOutput:
        del context, sample_context
        if history_states is None or history_actions is None:
            history_states, history_actions = self.default_history(states, actions)
        history = torch.cat(
            [
                history_states.reshape(history_states.shape[0], -1),
                history_actions.reshape(history_actions.shape[0], -1),
            ],
            dim=-1,
        )
        features = torch.cat([history, states, actions], dim=-1)
        backbone_delta, backbone_logvar = self.backbone(features).chunk(2, dim=-1)
        backbone_delta = backbone_delta.to(states.dtype)
        backbone_logvar = backbone_logvar.to(states.dtype).clamp(
            self.config.min_logvar,
            self.config.max_logvar,
        )
        symbolic_mean, symbolic_var, alpha = self._symbolic_distribution(
            states=states,
            actions=actions,
            history_states=history_states,
            history_actions=history_actions,
            features=features,
        )
        symbolic_feature = float(self.config.symbolic_delta_scale) * symbolic_mean
        symbolic_feature_var = (float(self.config.symbolic_delta_scale) ** 2) * symbolic_var
        conditioned = torch.cat(
            [
                features,
                alpha,
                symbolic_feature,
                symbolic_var.clamp_min(float(self.config.symbolic_var_floor)).sqrt(),
            ],
            dim=-1,
        )
        gate = torch.sigmoid(self.gate(conditioned)).to(states.dtype)
        if self.config.mode == "gr":
            symbolic_correction = gate * symbolic_feature
            correction_var = gate.pow(2) * symbolic_feature_var
        else:
            radius = (
                float(max(0.0, self.config.trust_scale))
                * symbolic_feature_var.clamp_min(float(self.config.symbolic_var_floor)).sqrt()
            )
            local = torch.tanh(self.trust_correction(conditioned).to(states.dtype)) * radius
            symbolic_correction = gate * (symbolic_feature + local)
            correction_var = gate.pow(2) * (symbolic_feature_var + radius.pow(2))

        delta = backbone_delta + symbolic_correction
        variance = (torch.exp(backbone_logvar) + correction_var).clamp_min(1e-8)
        logvar = variance.log().clamp(self.config.min_logvar, self.config.max_logvar)
        output = BaselineForwardOutput(mean=states + delta, logvar=logvar, latent=alpha)
        output.symbolic_mean = symbolic_mean
        output.symbolic_delta = symbolic_feature
        output.symbolic_var = symbolic_var
        output.alpha = alpha
        output.lean_gate = gate
        output.lean_correction = symbolic_correction
        output.backbone_delta = backbone_delta
        return output

    def nll(self, output: BaselineForwardOutput, targets: torch.Tensor) -> torch.Tensor:
        inv_var = torch.exp(-output.logvar)
        return 0.5 * ((targets - output.mean).pow(2) * inv_var + output.logvar).mean()

    def _symbolic_distribution(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        history_states: torch.Tensor,
        history_actions: torch.Tensor,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_effects = self.law_priors(states, actions, history_states, history_actions)
        alpha_logits = self.law_activation(features) + self.law_logit_bias.to(
            states.device,
            states.dtype,
        ).unsqueeze(0)
        alpha = torch.sigmoid(alpha_logits).to(states.dtype)
        confidence = self.law_confidence.to(states.device, states.dtype).clamp(0.0, 1.0)
        law_var = (
            self.law_prior_std.to(states.device, states.dtype)
            * self.context_scales.to(states.device, states.dtype)
        ).pow(2)
        law_var = law_var / confidence.clamp_min(0.05)
        mask = self.law_output_mask.to(states.device, states.dtype).unsqueeze(0)
        temperature = float(max(1e-3, self.config.symbolic_temperature))
        precision = (
            alpha
            * confidence.view(1, -1)
            / (temperature * law_var.clamp_min(float(self.config.symbolic_var_floor))).view(1, -1)
        ).unsqueeze(-1) * mask
        precision_sum = precision.sum(dim=1)
        symbolic_mean = torch.where(
            precision_sum > 1e-8,
            (precision * raw_effects).sum(dim=1) / precision_sum.clamp_min(1e-8),
            torch.zeros_like(states),
        ).to(states.dtype)
        symbolic_var = (1.0 / precision_sum.clamp_min(1e-8)).clamp(
            float(self.config.symbolic_var_floor),
            float(self.config.symbolic_var_ceiling),
        ).to(states.dtype)
        return symbolic_mean, symbolic_var, alpha


def fit_baseline_world_model(
    model: MLPWorldModel | PETSWorldModel | CaDMWorldModel | LEANWorldModel,
    transitions: MuJoCoTransitions,
    config: BaselineTrainerConfig,
    device: torch.device | str,
    control_templates: tuple[MechanismTemplate, ...],
    use_oracle_context: bool = False,
    context_supervision_weight: float = 0.0,
) -> list[dict[str, float]]:
    model.to(device)
    if use_oracle_context or context_supervision_weight > 0.0:
        transitions = align_contexts_to_templates(transitions, control_templates)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    autocast_enabled, autocast_dtype = _autocast_settings(config.precision, torch.device(device))
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=autocast_enabled and autocast_dtype == torch.float16,
    )
    control_weights_np = default_control_weights(transitions.state_dim, control_templates)
    control_weights = torch.tensor(control_weights_np, dtype=torch.float32, device=device)
    prepared = (
        prepare_duc_data(transitions, history_length=config.history_length, device=device)
        if config.preload_to_device
        else None
    )
    history: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        totals: list[float] = []
        nlls: list[float] = []
        controls: list[float] = []
        rolls: list[float] = []
        ctxs: list[float] = []
        batches = (
            iter_prepared_duc_batches(
                prepared,
                batch_size=config.batch_size,
                shuffle=True,
                seed=config.seed + epoch,
            )
            if prepared is not None
            else iter_duc_batches(
                transitions,
                batch_size=config.batch_size,
                history_length=config.history_length,
                shuffle=True,
                seed=config.seed + epoch,
                device=device,
            )
        )
        for batch in batches:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=torch.device(device).type,
                dtype=autocast_dtype,
                enabled=autocast_enabled,
            ):
                context = batch.contexts if use_oracle_context else None
                output = model(
                    batch.states,
                    batch.actions,
                    batch.history_states,
                    batch.history_actions,
                    context=context,
                )
                nll = model.nll(output, batch.next_states)
                batch_weights = control_weights.unsqueeze(0).expand_as(batch.states)
                control = weighted_mse(output.mean, batch.next_states, batch_weights)
                if context_supervision_weight > 0.0:
                    context_loss = _context_supervision_loss(output, batch)
                else:
                    context_loss = batch.states.new_zeros(())
                rollout = _rollout_loss_for_batch(
                    model=model,
                    transitions=transitions,
                    batch=batch,
                    horizon=config.rollout_horizon,
                    control_weights=control_weights,
                    device=device,
                    use_oracle_context=use_oracle_context,
                )
                total = (
                    nll
                    + config.control_weight * control
                    + config.rollout_weight * rollout
                    + float(context_supervision_weight) * context_loss
                )
            if scaler.is_enabled():
                scaler.scale(total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
            totals.append(float(total.detach().cpu()))
            nlls.append(float(nll.detach().cpu()))
            controls.append(float(control.detach().cpu()))
            rolls.append(float(rollout.detach().cpu()))
            ctxs.append(float(context_loss.detach().cpu()))
        history.append(
            {
                "epoch": float(epoch + 1),
                "loss": sum(totals) / max(1, len(totals)),
                "nll": sum(nlls) / max(1, len(nlls)),
                "control": sum(controls) / max(1, len(controls)),
                "rollout": sum(rolls) / max(1, len(rolls)),
                "context": sum(ctxs) / max(1, len(ctxs)),
            }
        )
    return history


def evaluate_baseline_world_model(
    model: MLPWorldModel | PETSWorldModel | CaDMWorldModel | LEANWorldModel,
    transitions: MuJoCoTransitions,
    device: torch.device | str,
    control_templates: tuple[MechanismTemplate, ...],
    batch_size: int = 512,
    history_length: int = 4,
    rollout_horizon: int = 5,
    use_oracle_context: bool = False,
) -> dict[str, float]:
    if use_oracle_context:
        transitions = align_contexts_to_templates(transitions, control_templates)
    return evaluate_world_model(
        model=model,
        transitions=transitions,
        device=device,
        control_templates=control_templates,
        batch_size=batch_size,
        history_length=history_length,
        rollout_horizon=rollout_horizon,
        use_oracle_context=use_oracle_context,
    )


def _rollout_loss_for_batch(
    model: MLPWorldModel | PETSWorldModel | CaDMWorldModel | LEANWorldModel,
    transitions: MuJoCoTransitions,
    batch: DUCBatch,
    horizon: int,
    control_weights: torch.Tensor,
    device: torch.device | str,
    use_oracle_context: bool = False,
) -> torch.Tensor:
    if horizon <= 1:
        return batch.states.new_zeros(())

    max_start = transitions.num_steps - horizon
    valid = batch.indices[batch.indices <= max_start]
    if transitions.dones is not None and len(valid) > 0:
        keep: list[int] = []
        for index in valid.detach().cpu().tolist():
            done_window = transitions.dones[index : index + horizon - 1]
            if not bool(done_window.any()):
                keep.append(index)
        valid = torch.tensor(keep, dtype=torch.long, device=device)
    if len(valid) == 0:
        return batch.states.new_zeros(())

    index_np = valid.detach().cpu().numpy()
    current = torch.tensor(transitions.states[index_np], dtype=torch.float32, device=device)
    history_states = torch.tensor(
        _history_for_indices(
            transitions.states,
            index_np,
            batch.history_states.shape[1],
            dones=transitions.dones,
        ),
        dtype=torch.float32,
        device=device,
    )
    history_actions = torch.tensor(
        _history_for_indices(
            transitions.actions,
            index_np,
            batch.history_actions.shape[1],
            dones=transitions.dones,
        ),
        dtype=torch.float32,
        device=device,
    )
    total = current.new_zeros(())
    for offset in range(horizon):
        step_indices = index_np + offset
        actions = torch.tensor(transitions.actions[step_indices], dtype=torch.float32, device=device)
        targets = torch.tensor(transitions.next_states[step_indices], dtype=torch.float32, device=device)
        context = None
        if use_oracle_context and transitions.contexts is not None:
            context = torch.tensor(transitions.contexts[step_indices], dtype=torch.float32, device=device)
        output = model(current, actions, history_states, history_actions, context=context)
        weights = control_weights.unsqueeze(0).expand_as(targets)
        total = total + weighted_mse(output.mean, targets, weights)
        current = output.mean
        history_states = torch.cat([history_states[:, 1:], current.unsqueeze(1)], dim=1)
        history_actions = torch.cat([history_actions[:, 1:], actions.unsqueeze(1)], dim=1)
    return total / float(horizon)


def _context_inputs(
    states: torch.Tensor,
    actions: torch.Tensor,
    context: torch.Tensor | None,
    context_dim: int,
) -> torch.Tensor:
    if context_dim <= 0:
        return torch.cat([states, actions], dim=-1)
    if context is None:
        context = states.new_zeros(states.shape[0], context_dim)
    if context.shape[-1] != context_dim:
        raise ValueError(
            f"context has dim {context.shape[-1]}, expected {context_dim}"
        )
    return torch.cat([states, actions, context], dim=-1)


def _context_supervision_loss(
    output: BaselineForwardOutput,
    batch: DUCBatch,
) -> torch.Tensor:
    if output.latent is None or batch.contexts is None:
        return batch.states.new_zeros(())
    if output.latent.shape[-1] != batch.contexts.shape[-1]:
        raise ValueError(
            "latent context dim must match supervised context dim: "
            f"{output.latent.shape[-1]} != {batch.contexts.shape[-1]}"
        )
    return (output.latent - batch.contexts).pow(2).mean()


def _autocast_settings(precision: str, device: torch.device) -> tuple[bool, torch.dtype]:
    if device.type != "cuda":
        return False, torch.float32
    if precision == "bf16":
        return True, torch.bfloat16
    if precision == "fp16":
        return True, torch.float16
    if precision in {"fp32", "none"}:
        return False, torch.float32
    raise ValueError("precision must be fp32, bf16, or fp16")


def _law_output_mask(
    templates: tuple[MechanismTemplate, ...],
    state_dim: int,
) -> torch.Tensor:
    mask = torch.zeros(len(templates), state_dim, dtype=torch.float32)
    for law_index, template in enumerate(templates):
        for output_index in template.output_indices:
            if 0 <= output_index < state_dim:
                mask[law_index, output_index] = 1.0
    return mask
