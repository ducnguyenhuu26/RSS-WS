from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from onelife.mujoco_dataset import MuJoCoTransitions

from .core import WorldModelForwardOutput, kl_normal_diag, mlp, weighted_mse
from .data import DUCBatch, iter_duc_batches, iter_prepared_duc_batches, prepare_duc_data
from .law_dsl import LawPriorBank
from .metrics import _history_for_indices, default_control_weights
from .templates import MechanismTemplate, prior_tensors


@dataclass(frozen=True)
class SimFuturesWorldModelConfig:
    state_dim: int
    action_dim: int
    templates: tuple[MechanismTemplate, ...]
    hidden_size: int = 256
    hidden_layers: int = 2
    history_length: int = 4
    min_logvar: float = -8.0
    max_logvar: float = 2.0
    symbolic_delta_scale: float = 0.20
    chart_count: int = 4


@dataclass(frozen=True)
class SimFuturesTrainerConfig:
    epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 1e-3
    history_length: int = 4
    beta_kl: float = 1e-3
    prior_kl_weight: float = 5e-4
    prior_path_weight: float = 0.25
    law_channel_weight: float = 0.10
    reward_weight: float = 0.10
    reliability_weight: float = 0.20
    control_weight: float = 0.05
    rollout_weight: float = 0.0
    rollout_horizon: int = 1
    posterior_update_interval: int = 1
    posterior_update_samples: int = 4096
    posterior_trust: float = 0.25
    posterior_temperature: float = 1.0
    utility_error_weight: float = 1.0
    utility_reward_gap_weight: float = 0.25
    utility_law_weight: float = 0.10
    seed: int = 0
    precision: str = "fp32"
    preload_to_device: bool = False


class SimFuturesWorldModel(nn.Module):
    """Future-simulated law posterior model for MuJoCo continuous control.

    The LLM/template bank is treated as an executable hyper-prior over law
    channels. A neural prior/posterior pair infers latent law validity, a clean
    conditional dynamics model predicts state deltas, and a reliability head
    learns whether a law-conditioned transition is useful for reward-seeking
    planning.
    """

    def __init__(self, config: SimFuturesWorldModelConfig) -> None:
        super().__init__()
        if not config.templates:
            raise ValueError("SimFuturesWorldModel requires at least one law template")
        self.config = config
        if config.chart_count <= 0:
            raise ValueError("chart_count must be positive")
        self.law_priors = LawPriorBank(config.templates, config.state_dim, config.action_dim)
        self.num_laws = len(config.templates)
        self.chart_count = int(config.chart_count)
        history_dim = config.history_length * (config.state_dim + config.action_dim)
        prior_input_dim = history_dim + config.state_dim + config.action_dim
        posterior_input_dim = prior_input_dim + config.state_dim + self.num_laws
        self.dyn_prior_encoder = mlp(
            input_dim=prior_input_dim,
            output_dim=2 * self.num_laws,
            hidden_size=config.hidden_size,
            hidden_layers=max(1, config.hidden_layers),
        )
        self.dyn_posterior_encoder = mlp(
            input_dim=posterior_input_dim,
            output_dim=2 * self.num_laws,
            hidden_size=config.hidden_size,
            hidden_layers=max(1, config.hidden_layers),
        )
        self.ctrl_prior_encoder = mlp(
            input_dim=prior_input_dim,
            output_dim=2 * self.num_laws,
            hidden_size=config.hidden_size,
            hidden_layers=max(1, config.hidden_layers),
        )
        self.ctrl_posterior_encoder = mlp(
            input_dim=posterior_input_dim,
            output_dim=2 * self.num_laws,
            hidden_size=config.hidden_size,
            hidden_layers=max(1, config.hidden_layers),
        )
        chart_input_dim = (
            prior_input_dim
            + self.num_laws
            + config.state_dim
        )
        self.chart_gate = mlp(
            input_dim=prior_input_dim,
            output_dim=self.chart_count,
            hidden_size=config.hidden_size,
            hidden_layers=max(1, config.hidden_layers - 1),
        )
        self.dynamics_experts = nn.ModuleList(
            [
                mlp(
                    input_dim=chart_input_dim,
                    output_dim=2 * config.state_dim,
                    hidden_size=config.hidden_size,
                    hidden_layers=config.hidden_layers,
                )
                for _ in range(self.chart_count)
            ]
        )
        self.reward_head = mlp(
            input_dim=config.state_dim + config.action_dim + config.state_dim + self.num_laws,
            output_dim=1,
            hidden_size=config.hidden_size,
            hidden_layers=max(1, config.hidden_layers - 1),
        )
        self.law_observer = mlp(
            input_dim=self.num_laws,
            output_dim=self.num_laws,
            hidden_size=config.hidden_size,
            hidden_layers=max(1, config.hidden_layers - 1),
        )
        planning_input_dim = (
            config.state_dim
            + config.action_dim
            + self.num_laws
            + 2 * config.state_dim
        )
        self.reliability_head = mlp(
            input_dim=planning_input_dim,
            output_dim=1,
            hidden_size=config.hidden_size,
            hidden_layers=max(1, config.hidden_layers - 1),
        )

        prior_mean, prior_std, scales, confidences = prior_tensors(config.templates)
        self.register_buffer("law_prior_mean", prior_mean)
        self.register_buffer("law_prior_std", prior_std.clamp_min(0.05))
        self.register_buffer("context_scales", scales)
        self.register_buffer("prior_confidence", confidences)
        init_logits = torch.logit(confidences.clamp(1e-3, 1.0 - 1e-3))
        self.register_buffer("law_posterior_logits", init_logits)
        self.register_buffer("prior_gate", confidences.clone())
        self.register_buffer("data_confidence", confidences.clone())
        self.register_buffer("reward_sensitivity", torch.ones(config.state_dim))
        self.register_buffer("_residual_scale", torch.tensor(1.0, dtype=torch.float32))
        self._planning_mode = False
        self.unknown_indices = tuple(
            index
            for index, template in enumerate(config.templates)
            if template.timescale == "unknown" or template.name == "unknown"
        )

    @property
    def context_dim(self) -> int:
        return self.num_laws

    @property
    def prior_beta(self) -> torch.Tensor:
        return self.law_posterior_probs.clamp_min(0.05)

    @property
    def law_posterior_probs(self) -> torch.Tensor:
        return torch.sigmoid(self.law_posterior_logits)

    @property
    def effective_prior_confidence(self) -> torch.Tensor:
        return (self.prior_gate * self.data_confidence).clamp(0.0, 1.0)

    @torch.no_grad()
    def set_reward_sensitivity(self, weights: torch.Tensor) -> None:
        weights = weights.to(device=self.reward_sensitivity.device, dtype=self.reward_sensitivity.dtype)
        if weights.shape != self.reward_sensitivity.shape:
            raise ValueError(
                f"reward sensitivity has shape {tuple(weights.shape)}, "
                f"expected {tuple(self.reward_sensitivity.shape)}"
            )
        self.reward_sensitivity.copy_(weights.clamp(0.1, 20.0))

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
        self.prior_gate.copy_(gate.clamp(0.0, 1.0))
        self.data_confidence.copy_(data_confidence.clamp(0.0, 1.0))
        if beta is not None:
            beta = beta.to(device=self.law_posterior_logits.device, dtype=self.law_posterior_logits.dtype)
            posterior = (self.law_posterior_probs * beta.clamp(0.05, 8.0)).clamp(1e-3, 1.0 - 1e-3)
            self.law_posterior_logits.copy_(torch.logit(posterior))

    def set_residual_scale(self, value: float) -> None:
        self._residual_scale.fill_(float(max(0.0, min(1.0, value))))

    def set_planning_mode(self, enabled: bool) -> None:
        self._planning_mode = bool(enabled)

    @property
    def planning_mode(self) -> bool:
        return bool(self._planning_mode)

    @torch.no_grad()
    def update_law_posterior(
        self,
        evidence: torch.Tensor,
        trust: float,
        temperature: float,
    ) -> None:
        evidence = evidence.to(
            device=self.law_posterior_logits.device,
            dtype=self.law_posterior_logits.dtype,
        )
        if evidence.shape != self.law_posterior_logits.shape:
            raise ValueError(
                f"posterior evidence shape {tuple(evidence.shape)} does not match "
                f"{tuple(self.law_posterior_logits.shape)}"
            )
        evidence = evidence.nan_to_num(0.0)
        evidence = evidence - evidence.mean()
        evidence = evidence / evidence.std().clamp_min(1e-6)
        temp = max(1e-6, float(temperature))
        target_logits = torch.logit(self.prior_confidence.clamp(1e-3, 1.0 - 1e-3)) + evidence / temp
        trust = float(max(0.0, min(1.0, trust)))
        self.law_posterior_logits.copy_(
            (1.0 - trust) * self.law_posterior_logits + trust * target_logits
        )

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
        next_states: torch.Tensor | None = None,
    ) -> WorldModelForwardOutput:
        if history_states is None or history_actions is None:
            history_states, history_actions = self.default_history(states, actions)
        raw_prior_effects = self.law_priors(
            states,
            actions,
            history_states=history_states,
            history_actions=history_actions,
        ).to(states.dtype)
        target_delta = None if next_states is None else (next_states - states).to(states.dtype)
        law_targets = (
            law_channel_targets(raw_prior_effects, target_delta, None)
            if target_delta is not None
            else states.new_zeros(states.shape[0], self.num_laws)
        )
        utility_law_targets = (
            law_channel_targets(raw_prior_effects, target_delta, self.reward_sensitivity)
            if target_delta is not None
            else states.new_zeros(states.shape[0], self.num_laws)
        )

        prior_features = torch.cat(
            [
                history_states.reshape(states.shape[0], -1),
                history_actions.reshape(states.shape[0], -1),
                states,
                actions,
            ],
            dim=-1,
        )
        prior_mean, prior_logvar = self.dyn_prior_encoder(prior_features).chunk(2, dim=-1)
        ctrl_prior_mean, ctrl_prior_logvar = self.ctrl_prior_encoder(prior_features).chunk(2, dim=-1)
        posterior_mean = prior_mean
        posterior_logvar = prior_logvar
        ctrl_posterior_mean = ctrl_prior_mean
        ctrl_posterior_logvar = ctrl_prior_logvar
        if next_states is not None:
            posterior_features = torch.cat([prior_features, target_delta, law_targets], dim=-1)
            ctrl_posterior_features = torch.cat([prior_features, target_delta, utility_law_targets], dim=-1)
            posterior_mean, posterior_logvar = self.dyn_posterior_encoder(posterior_features).chunk(2, dim=-1)
            ctrl_posterior_mean, ctrl_posterior_logvar = self.ctrl_posterior_encoder(ctrl_posterior_features).chunk(2, dim=-1)
        posterior_logvar = posterior_logvar.clamp(-8.0, 4.0)
        prior_logvar = prior_logvar.clamp(-8.0, 4.0)
        ctrl_posterior_logvar = ctrl_posterior_logvar.clamp(-8.0, 4.0)
        ctrl_prior_logvar = ctrl_prior_logvar.clamp(-8.0, 4.0)

        if context is not None:
            alpha = context.clamp(0.0, 1.0)
            alpha_mean = alpha
            alpha_ctrl = alpha
            alpha_ctrl_mean = alpha
        else:
            raw = posterior_mean
            if sample_context and self.training:
                raw = raw + torch.exp(0.5 * posterior_logvar) * torch.randn_like(raw)
            dyn_bias = torch.logit(self.prior_gate.clamp(1e-3, 1.0 - 1e-3)).to(raw.device, raw.dtype).unsqueeze(0)
            alpha = torch.sigmoid(raw + dyn_bias)
            alpha_mean = torch.sigmoid(posterior_mean + dyn_bias)
            raw_ctrl = ctrl_posterior_mean
            if sample_context and self.training:
                raw_ctrl = raw_ctrl + torch.exp(0.5 * ctrl_posterior_logvar) * torch.randn_like(raw_ctrl)
            ctrl_bias = self.law_posterior_logits.to(raw_ctrl.device, raw_ctrl.dtype).unsqueeze(0)
            alpha_ctrl = torch.sigmoid(raw_ctrl + ctrl_bias)
            alpha_ctrl_mean = torch.sigmoid(ctrl_posterior_mean + ctrl_bias)

        gated_prior = raw_prior_effects * self.prior_gate.to(states.device, states.dtype).view(1, -1, 1)
        symbolic_delta = torch.einsum("bk,bkd->bd", alpha, gated_prior).to(states.dtype)
        symbolic_feature = float(self.config.symbolic_delta_scale) * symbolic_delta
        chart_logits = self.chart_gate(prior_features)
        chart_probs = torch.softmax(chart_logits, dim=-1).to(states.dtype)
        chart_input = torch.cat([prior_features, alpha, symbolic_feature], dim=-1)
        expert_means: list[torch.Tensor] = []
        expert_logvars: list[torch.Tensor] = []
        for expert in self.dynamics_experts:
            delta_i, logvar_i = expert(chart_input).chunk(2, dim=-1)
            expert_means.append(delta_i.to(states.dtype))
            expert_logvars.append(logvar_i.clamp(self.config.min_logvar, self.config.max_logvar).to(states.dtype))
        expert_delta = torch.stack(expert_means, dim=1)
        expert_logvar = torch.stack(expert_logvars, dim=1)
        chart_weights = chart_probs.unsqueeze(-1)
        base_delta = (chart_weights * expert_delta).sum(dim=1)
        expert_var = torch.exp(expert_logvar)
        second_moment = (chart_weights * (expert_var + expert_delta.pow(2))).sum(dim=1)
        mixture_var = (second_moment - base_delta.pow(2)).clamp_min(1e-8)
        logvar = mixture_var.log().clamp(self.config.min_logvar, self.config.max_logvar)
        prior_delta = symbolic_feature
        prediction_mean = states + base_delta
        planning_delta = states.new_zeros(states.shape)
        planning_mean = prediction_mean
        mean = prediction_mean

        reward_input = torch.cat([states.detach(), actions.detach(), prediction_mean.detach(), alpha_ctrl.detach()], dim=-1)
        planning_input = torch.cat(
            [
                states.detach(),
                actions.detach(),
                alpha_ctrl.detach(),
                base_delta.detach(),
                symbolic_feature.detach(),
            ],
            dim=-1,
        )
        planning_reward_input = torch.cat([states.detach(), actions.detach(), planning_mean.detach(), alpha_ctrl.detach()], dim=-1)
        reward_pred = self.reward_head(reward_input).squeeze(-1).to(states.dtype)
        planning_reward_pred = self.reward_head(planning_reward_input).squeeze(-1).to(states.dtype)
        law_channel_pred = torch.sigmoid(self.law_observer(alpha)).to(states.dtype)
        planning_bonus = self.reliability_head(planning_input).squeeze(-1).to(states.dtype)

        zero_effects = raw_prior_effects.new_zeros(raw_prior_effects.shape)
        output = WorldModelForwardOutput(
            mean=mean,
            prediction_mean=prediction_mean,
            planning_mean=planning_mean,
            logvar=logvar.clamp(self.config.min_logvar, self.config.max_logvar),
            effects=gated_prior,
            prior_effects=gated_prior,
            residual_effects=zero_effects,
            raw_prior_effects=raw_prior_effects,
            raw_residual_effects=zero_effects,
            alpha=alpha,
            alpha_mean=alpha_mean,
            posterior_mean=posterior_mean,
            posterior_logvar=posterior_logvar,
            base_delta=base_delta,
            context_delta=states.new_zeros(states.shape),
            prior_delta=prior_delta,
            residual_delta=states.new_zeros(states.shape),
            mechanism_delta=base_delta,
            proposed_mechanism_delta=symbolic_feature,
            mechanism_mix=chart_probs,
            planning_delta=planning_delta,
            prior_beta=self.prior_beta.to(states.device, states.dtype),
            residual_scale=self._residual_scale.to(states.device, states.dtype),
            prior_gate=self.prior_gate.to(states.device, states.dtype),
            data_confidence=self.data_confidence.to(states.device, states.dtype),
            reward_pred=reward_pred,
            planning_reward_pred=planning_reward_pred,
        )
        output.prior_mean = prior_mean
        output.prior_logvar = prior_logvar
        output.ctrl_prior_mean = ctrl_prior_mean
        output.ctrl_prior_logvar = ctrl_prior_logvar
        output.ctrl_posterior_mean = ctrl_posterior_mean
        output.ctrl_posterior_logvar = ctrl_posterior_logvar
        output.alpha_dyn = alpha
        output.alpha_dyn_mean = alpha_mean
        output.alpha_ctrl = alpha_ctrl
        output.alpha_ctrl_mean = alpha_ctrl_mean
        output.chart_probs = chart_probs
        output.chart_logits = chart_logits
        output.law_channel_pred = law_channel_pred
        output.law_channel_targets = law_targets
        output.utility_law_targets = utility_law_targets
        output.planning_bonus = planning_bonus
        output.symbolic_delta = symbolic_feature
        return output

    def nll(self, output: WorldModelForwardOutput, targets: torch.Tensor) -> torch.Tensor:
        inv_var = torch.exp(-output.logvar)
        return 0.5 * ((targets - output.mean).pow(2) * inv_var + output.logvar).mean()


def fit_simfutures_world_model(
    model: SimFuturesWorldModel,
    transitions: MuJoCoTransitions,
    config: SimFuturesTrainerConfig,
    device: torch.device | str,
) -> list[dict[str, float]]:
    model.to(device)
    reward_sensitivity = estimate_reward_sensitivity(
        transitions=transitions,
        device=device,
        scale=4.0,
        max_weight=6.0,
    )
    model.set_reward_sensitivity(reward_sensitivity)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    autocast_enabled, autocast_dtype = _autocast_settings(config.precision, torch.device(device))
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=autocast_enabled and autocast_dtype == torch.float16,
    )
    control_weights_np = default_control_weights(transitions.state_dim, model.config.templates)
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
        kls: list[float] = []
        ctrl_kls: list[float] = []
        prior_kls: list[float] = []
        prior_path_losses: list[float] = []
        law_losses: list[float] = []
        reward_losses: list[float] = []
        reliability_losses: list[float] = []
        control_losses: list[float] = []
        rollout_losses: list[float] = []
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
                output = model(
                    batch.states,
                    batch.actions,
                    batch.history_states,
                    batch.history_actions,
                    sample_context=False,
                    next_states=batch.next_states,
                )
                nll = model.nll(output, batch.next_states)
                kl = kl_normal_diag(
                    output.posterior_mean,
                    output.posterior_logvar,
                    output.prior_mean.detach(),
                    torch.exp(0.5 * output.prior_logvar.detach()).clamp_min(1e-4),
                )
                ctrl_kl = kl_normal_diag(
                    output.ctrl_posterior_mean,
                    output.ctrl_posterior_logvar,
                    output.ctrl_prior_mean.detach(),
                    torch.exp(0.5 * output.ctrl_prior_logvar.detach()).clamp_min(1e-4),
                )
                prior_kl = kl_normal_diag(
                    output.prior_mean,
                    output.prior_logvar,
                    model.law_prior_mean.to(output.prior_mean.device),
                    model.law_prior_std.to(output.prior_mean.device),
                )
                prior_output = model(
                    batch.states,
                    batch.actions,
                    batch.history_states,
                    batch.history_actions,
                    sample_context=False,
                    next_states=None,
                )
                prior_path = model.nll(prior_output, batch.next_states)
                law_loss = (output.law_channel_pred - output.law_channel_targets.detach()).pow(2).mean()
                if batch.rewards is None:
                    reward_loss = batch.states.new_zeros(())
                    reliability_loss = batch.states.new_zeros(())
                else:
                    reward_loss = (output.reward_pred - batch.rewards).pow(2).mean()
                    utility = wake_utility_targets(
                        output=output,
                        batch=batch,
                        control_weights=control_weights,
                        config=config,
                    )
                    reliability_loss = (output.planning_bonus - utility.detach()).pow(2).mean()
                batch_weights = control_weights.unsqueeze(0).expand_as(batch.states)
                control = weighted_mse(output.mean, batch.next_states, batch_weights)
                rollout = _rollout_loss_for_batch(
                    model=model,
                    transitions=transitions,
                    batch=batch,
                    horizon=config.rollout_horizon,
                    control_weights=control_weights,
                    device=device,
                )
                total = (
                    nll
                    + config.beta_kl * (kl + ctrl_kl)
                    + config.prior_kl_weight * prior_kl
                    + config.prior_path_weight * prior_path
                    + config.law_channel_weight * law_loss
                    + config.reward_weight * reward_loss
                    + config.reliability_weight * reliability_loss
                    + config.control_weight * control
                    + config.rollout_weight * rollout
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
            kls.append(float(kl.detach().cpu()))
            ctrl_kls.append(float(ctrl_kl.detach().cpu()))
            prior_kls.append(float(prior_kl.detach().cpu()))
            prior_path_losses.append(float(prior_path.detach().cpu()))
            law_losses.append(float(law_loss.detach().cpu()))
            reward_losses.append(float(reward_loss.detach().cpu()))
            reliability_losses.append(float(reliability_loss.detach().cpu()))
            control_losses.append(float(control.detach().cpu()))
            rollout_losses.append(float(rollout.detach().cpu()))

        posterior_stats = {}
        if (
            config.posterior_update_interval > 0
            and (epoch + 1) % config.posterior_update_interval == 0
        ):
            posterior_stats = calibrate_law_posterior(
                model=model,
                transitions=transitions,
                config=config,
                device=device,
            )
        record = {
            "epoch": float(epoch + 1),
            "loss": float(np.mean(totals)) if totals else 0.0,
            "nll": float(np.mean(nlls)) if nlls else 0.0,
            "kl": float(np.mean(kls)) if kls else 0.0,
            "ctrl_kl": float(np.mean(ctrl_kls)) if ctrl_kls else 0.0,
            "prior_kl": float(np.mean(prior_kls)) if prior_kls else 0.0,
            "prior_path": float(np.mean(prior_path_losses)) if prior_path_losses else 0.0,
            "law_channel": float(np.mean(law_losses)) if law_losses else 0.0,
            "reward": float(np.mean(reward_losses)) if reward_losses else 0.0,
            "reliability": float(np.mean(reliability_losses)) if reliability_losses else 0.0,
            "control": float(np.mean(control_losses)) if control_losses else 0.0,
            "rollout": float(np.mean(rollout_losses)) if rollout_losses else 0.0,
            "posterior_entropy": law_posterior_entropy(model),
            "posterior_mean": float(model.law_posterior_probs.mean().detach().cpu()),
        }
        record.update(posterior_stats)
        history.append(record)
    return history


@torch.no_grad()
def calibrate_law_posterior(
    model: SimFuturesWorldModel,
    transitions: MuJoCoTransitions,
    config: SimFuturesTrainerConfig,
    device: torch.device | str,
) -> dict[str, float]:
    if transitions.rewards is None or transitions.num_steps <= 0:
        return {}
    sample_count = int(max(1, min(config.posterior_update_samples, transitions.num_steps)))
    rng = np.random.default_rng(config.seed + 71_771)
    indices = (
        np.sort(rng.choice(transitions.num_steps, size=sample_count, replace=False))
        if sample_count < transitions.num_steps
        else np.arange(transitions.num_steps)
    )
    states = torch.tensor(transitions.states[indices], dtype=torch.float32, device=device)
    actions = torch.tensor(transitions.actions[indices], dtype=torch.float32, device=device)
    next_states = torch.tensor(transitions.next_states[indices], dtype=torch.float32, device=device)
    rewards = torch.tensor(transitions.rewards[indices], dtype=torch.float32, device=device)
    history_states = torch.tensor(
        _history_for_indices(transitions.states, indices, config.history_length, dones=transitions.dones),
        dtype=torch.float32,
        device=device,
    )
    history_actions = torch.tensor(
        _history_for_indices(transitions.actions, indices, config.history_length, dones=transitions.dones),
        dtype=torch.float32,
        device=device,
    )
    was_training = model.training
    model.eval()
    output = model(
        states,
        actions,
        history_states,
        history_actions,
        sample_context=False,
        next_states=next_states,
    )
    utility = wake_utility_targets(
        output=output,
        batch=DUCBatch(
            indices=torch.tensor(indices, dtype=torch.long, device=device),
            states=states,
            actions=actions,
            next_states=next_states,
            history_states=history_states,
            history_actions=history_actions,
            rewards=rewards,
        ),
        control_weights=torch.tensor(
            default_control_weights(transitions.state_dim, model.config.templates),
            dtype=torch.float32,
            device=device,
        ),
        config=config,
    )
    centered_utility = utility - utility.mean()
    alpha_ctrl = getattr(output, "alpha_ctrl_mean", output.alpha_mean)
    evidence = (alpha_ctrl.detach() * centered_utility.unsqueeze(-1)).mean(dim=0)
    model.update_law_posterior(
        evidence=evidence,
        trust=config.posterior_trust,
        temperature=config.posterior_temperature,
    )
    if was_training:
        model.train()
    return {
        "law_posterior_evidence_mean": float(evidence.mean().detach().cpu()),
        "law_posterior_evidence_std": float(evidence.std().detach().cpu()),
        "law_posterior_entropy": law_posterior_entropy(model),
    }


def law_channel_targets(
    raw_prior_effects: torch.Tensor,
    target_delta: torch.Tensor | None,
    state_weights: torch.Tensor | None,
) -> torch.Tensor:
    batch, num_laws, _ = raw_prior_effects.shape
    if target_delta is None:
        return raw_prior_effects.new_zeros(batch, num_laws)
    if state_weights is None:
        weights = raw_prior_effects.new_ones(1, 1, raw_prior_effects.shape[-1])
    else:
        weights = state_weights.to(raw_prior_effects.device, raw_prior_effects.dtype).view(1, 1, -1)
    prior = raw_prior_effects.float() * weights.float().sqrt()
    target = target_delta.float().unsqueeze(1) * weights.float().sqrt()
    dot = (prior * target).sum(dim=-1)
    prior_norm = prior.pow(2).sum(dim=-1).sqrt().clamp_min(1e-8)
    target_norm = target.pow(2).sum(dim=-1).sqrt().clamp_min(1e-8)
    cosine = (dot / (prior_norm * target_norm)).clamp(-1.0, 1.0)
    scale = (dot.abs() / prior_norm.pow(2).clamp_min(1e-8)).clamp(0.0, 4.0)
    return (cosine.relu() * torch.tanh(scale)).clamp(0.0, 1.0).to(raw_prior_effects.dtype)


def wake_utility_targets(
    output: WorldModelForwardOutput,
    batch: DUCBatch,
    control_weights: torch.Tensor,
    config: SimFuturesTrainerConfig,
) -> torch.Tensor:
    if batch.rewards is None:
        return batch.states.new_zeros(batch.states.shape[0])
    weights = control_weights.to(batch.states.device, batch.states.dtype).unsqueeze(0)
    state_error = ((output.prediction_mean - batch.next_states).pow(2) * weights).mean(dim=-1)
    reward_error = (output.reward_pred - batch.rewards).abs()
    law_targets = getattr(output, "utility_law_targets", output.law_channel_targets)
    law_error = (output.law_channel_pred - law_targets.detach()).pow(2).mean(dim=-1)
    rewards = batch.rewards
    reward_score = (rewards - rewards.mean()) / rewards.std().clamp_min(1e-6)
    error_score = state_error / state_error.detach().mean().clamp_min(1e-6)
    reward_gap_score = reward_error / reward_error.detach().mean().clamp_min(1e-6)
    law_score = law_error / law_error.detach().mean().clamp_min(1e-6)
    return (
        reward_score
        - float(config.utility_error_weight) * error_score
        - float(config.utility_reward_gap_weight) * reward_gap_score
        - float(config.utility_law_weight) * law_score
    ).clamp(-10.0, 10.0)


def law_posterior_entropy(model: SimFuturesWorldModel) -> float:
    probs = model.law_posterior_probs.detach().clamp(1e-6, 1.0 - 1e-6)
    entropy = -(probs * probs.log() + (1.0 - probs) * (1.0 - probs).log())
    return float(entropy.mean().cpu())


def estimate_reward_sensitivity(
    transitions: MuJoCoTransitions,
    device: torch.device | str,
    scale: float,
    max_weight: float,
) -> torch.Tensor:
    if transitions.rewards is None:
        return torch.ones(transitions.state_dim, dtype=torch.float32, device=device)
    rewards = torch.tensor(transitions.rewards, dtype=torch.float32, device=device)
    if rewards.numel() < 2 or float(rewards.std().detach().cpu()) <= 1e-8:
        return torch.ones(transitions.state_dim, dtype=torch.float32, device=device)
    states = torch.tensor(transitions.states, dtype=torch.float32, device=device)
    next_states = torch.tensor(transitions.next_states, dtype=torch.float32, device=device)
    delta = next_states - states
    reward_centered = rewards - rewards.mean()
    reward_std = reward_centered.std().clamp_min(1e-6)
    sensitivity = torch.maximum(
        _absolute_correlation(next_states, reward_centered, reward_std),
        _absolute_correlation(delta, reward_centered, reward_std),
    )
    if float(sensitivity.max().detach().cpu()) > 1e-8:
        sensitivity = sensitivity / sensitivity.max().clamp_min(1e-8)
    return (1.0 + float(scale) * sensitivity).clamp(1.0, float(max_weight))


def _absolute_correlation(
    values: torch.Tensor,
    reward_centered: torch.Tensor,
    reward_std: torch.Tensor,
) -> torch.Tensor:
    centered = values - values.mean(dim=0, keepdim=True)
    std = centered.std(dim=0).clamp_min(1e-6)
    covariance = (centered * reward_centered.unsqueeze(-1)).mean(dim=0)
    return (covariance / (std * reward_std)).abs().nan_to_num(0.0)


def _rollout_loss_for_batch(
    model: SimFuturesWorldModel,
    transitions: MuJoCoTransitions,
    batch: DUCBatch,
    horizon: int,
    control_weights: torch.Tensor,
    device: torch.device | str,
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
        _history_for_indices(transitions.states, index_np, batch.history_states.shape[1], dones=transitions.dones),
        dtype=torch.float32,
        device=device,
    )
    history_actions = torch.tensor(
        _history_for_indices(transitions.actions, index_np, batch.history_actions.shape[1], dones=transitions.dones),
        dtype=torch.float32,
        device=device,
    )
    total = current.new_zeros(())
    for offset in range(horizon):
        step_indices = index_np + offset
        actions = torch.tensor(transitions.actions[step_indices], dtype=torch.float32, device=device)
        targets = torch.tensor(transitions.next_states[step_indices], dtype=torch.float32, device=device)
        output = model(current, actions, history_states, history_actions, sample_context=False)
        weights = control_weights.unsqueeze(0).expand_as(targets)
        total = total + weighted_mse(output.mean, targets, weights)
        current = output.mean
        history_states = torch.cat([history_states[:, 1:], current.unsqueeze(1)], dim=1)
        history_actions = torch.cat([history_actions[:, 1:], actions.unsqueeze(1)], dim=1)
    return total / float(horizon)


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
