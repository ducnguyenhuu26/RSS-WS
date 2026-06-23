from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from onelife.mujoco_dataset import MuJoCoTransitions

from .data import (
    DUCBatch,
    align_contexts_to_templates,
    iter_duc_batches,
    iter_prepared_duc_batches,
    prepare_duc_data,
)
from .losses import DUCLossConfig, compute_duc_loss, weighted_mse
from .metrics import _history_for_indices, default_control_weights
from .model import DUCWorldModel


@dataclass(frozen=True)
class DUCTrainerConfig:
    epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 1e-3
    history_length: int = 4
    beta_kl: float = 1e-3
    context_weight: float = 1.0
    residual_weight: float = 0.0
    control_weight: float = 0.0
    rollout_weight: float = 0.0
    rollout_horizon: int = 1
    orth_weight: float = 0.0
    sparse_weight: float = 0.0
    unknown_weight: float = 0.0
    trust_region_weight: float = 0.2
    trust_region_delta_min: float = 0.15
    trust_region_delta_range: float = 0.75
    prior_beta_weight: float = 5e-4
    residual_warmup_fraction: float = 0.25
    prior_validation: bool = True
    prior_validation_min_gate: float = 0.02
    prior_validation_temperature: float = 0.15
    prior_validation_max_samples: int = 4096
    prior_validation_beta_min: float = 0.05
    prior_validation_beta_max: float = 5.0
    teacher_force_context: bool = True
    seed: int = 0
    precision: str = "fp32"
    preload_to_device: bool = False


def fit_duc_world_model(
    model: DUCWorldModel,
    transitions: MuJoCoTransitions,
    config: DUCTrainerConfig,
    device: torch.device | str,
) -> list[dict[str, float]]:
    model.to(device)
    transitions = align_contexts_to_templates(transitions, model.config.templates)
    if config.prior_validation:
        calibrate_prior_validation(
            model=model,
            transitions=transitions,
            config=config,
            device=device,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    autocast_enabled, autocast_dtype = _autocast_settings(config.precision, torch.device(device))
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=autocast_enabled and autocast_dtype == torch.float16,
    )
    loss_config = DUCLossConfig(
        beta_kl=config.beta_kl,
        context_weight=config.context_weight,
        residual_weight=config.residual_weight,
        control_weight=config.control_weight,
        orth_weight=config.orth_weight,
        sparse_weight=config.sparse_weight,
        unknown_weight=config.unknown_weight,
        trust_region_weight=config.trust_region_weight,
        trust_region_delta_min=config.trust_region_delta_min,
        trust_region_delta_range=config.trust_region_delta_range,
        prior_beta_weight=config.prior_beta_weight,
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
        residual_scale = residual_scale_for_epoch(epoch, config.epochs, config.residual_warmup_fraction)
        model.set_residual_scale(residual_scale)
        totals: list[float] = []
        nlls: list[float] = []
        kls: list[float] = []
        ctxs: list[float] = []
        residuals: list[float] = []
        trusts: list[float] = []
        betas: list[float] = []
        rolls: list[float] = []
        unknowns: list[float] = []
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
            context = batch.contexts if config.teacher_force_context else None
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
                    context=context,
                )
                batch_weights = control_weights.unsqueeze(0).expand_as(batch.states)
                loss = compute_duc_loss(
                    model=model,
                    output=output,
                    targets=batch.next_states,
                    context_targets=batch.contexts,
                    config=loss_config,
                    control_weights=batch_weights,
                )
                rollout = _rollout_loss_for_batch(
                    model=model,
                    transitions=transitions,
                    batch=batch,
                    horizon=config.rollout_horizon,
                    control_weights=control_weights,
                    teacher_force_context=config.teacher_force_context,
                    device=device,
                )
                total = loss.total + config.rollout_weight * rollout
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
            nlls.append(float(loss.nll.cpu()))
            kls.append(float(loss.kl.cpu()))
            ctxs.append(float(loss.context.cpu()))
            residuals.append(float(loss.residual.cpu()))
            trusts.append(float(loss.trust_region.cpu()))
            betas.append(float(loss.prior_beta.cpu()))
            rolls.append(float(rollout.detach().cpu()))
            unknowns.append(float(loss.unknown.cpu()))
        history.append(
            {
                "epoch": float(epoch + 1),
                "loss": sum(totals) / max(1, len(totals)),
                "nll": sum(nlls) / max(1, len(nlls)),
                "kl": sum(kls) / max(1, len(kls)),
                "context": sum(ctxs) / max(1, len(ctxs)),
                "residual": sum(residuals) / max(1, len(residuals)),
                "trust_region": sum(trusts) / max(1, len(trusts)),
                "prior_beta": sum(betas) / max(1, len(betas)),
                "residual_scale": residual_scale,
                "rollout": sum(rolls) / max(1, len(rolls)),
                "unknown": sum(unknowns) / max(1, len(unknowns)),
            }
        )
    model.set_residual_scale(1.0)
    return history


@torch.no_grad()
def calibrate_prior_validation(
    model: DUCWorldModel,
    transitions: MuJoCoTransitions,
    config: DUCTrainerConfig,
    device: torch.device | str,
) -> dict[str, float]:
    """Validate declarative law priors against observed transition deltas.

    LLM/template laws are useful only if their direction matches the data. This
    routine measures each prior effect before training and turns it into:

    - a gate that scales the prior contribution in the forward pass;
    - a data confidence used by residual/trust-region regularizers;
    - a bounded beta initializer fitted by least squares.

    Bad laws therefore become weak hints instead of hard constraints.
    """

    num_templates = len(model.config.templates)
    if transitions.num_steps <= 0 or num_templates == 0:
        return {}

    sample_count = int(max(1, min(config.prior_validation_max_samples, transitions.num_steps)))
    rng = np.random.default_rng(config.seed + 17_129)
    if sample_count < transitions.num_steps:
        indices = np.sort(rng.choice(transitions.num_steps, size=sample_count, replace=False))
    else:
        indices = np.arange(transitions.num_steps)

    states = torch.tensor(transitions.states[indices], dtype=torch.float32, device=device)
    actions = torch.tensor(transitions.actions[indices], dtype=torch.float32, device=device)
    next_states = torch.tensor(transitions.next_states[indices], dtype=torch.float32, device=device)
    history_states = torch.tensor(
        _history_for_indices(
            transitions.states,
            indices,
            config.history_length,
            dones=transitions.dones,
        ),
        dtype=torch.float32,
        device=device,
    )
    history_actions = torch.tensor(
        _history_for_indices(
            transitions.actions,
            indices,
            config.history_length,
            dones=transitions.dones,
        ),
        dtype=torch.float32,
        device=device,
    )
    contexts = None
    if transitions.contexts is not None:
        contexts = torch.tensor(transitions.contexts[indices], dtype=torch.float32, device=device)

    was_training = model.training
    model.eval()
    raw_priors = model.law_priors(
        states,
        actions,
        history_states=history_states,
        history_actions=history_actions,
    ).float()
    target_delta = (next_states - states).float()

    gates: list[float] = []
    confidences: list[float] = []
    betas: list[float] = []
    scores: list[float] = []
    for mechanism_index, template in enumerate(model.config.templates):
        gate, confidence, beta, score = _validate_single_prior(
            raw_prior=raw_priors[:, mechanism_index],
            target_delta=target_delta,
            context=contexts[:, mechanism_index] if contexts is not None else None,
            output_indices=template.output_indices,
            law_type=template.law_type,
            min_gate=config.prior_validation_min_gate,
            temperature=config.prior_validation_temperature,
            beta_min=config.prior_validation_beta_min,
            beta_max=config.prior_validation_beta_max,
        )
        gates.append(gate)
        confidences.append(confidence)
        betas.append(beta)
        scores.append(score)

    model.set_prior_validation(
        gate=torch.tensor(gates, dtype=torch.float32, device=device),
        data_confidence=torch.tensor(confidences, dtype=torch.float32, device=device),
        beta=torch.tensor(betas, dtype=torch.float32, device=device),
    )
    if was_training:
        model.train()
    return {
        "prior_gate_mean": float(np.mean(gates)),
        "data_confidence_mean": float(np.mean(confidences)),
        "prior_validation_score_mean": float(np.mean(scores)),
    }


def _validate_single_prior(
    raw_prior: torch.Tensor,
    target_delta: torch.Tensor,
    context: torch.Tensor | None,
    output_indices: tuple[int, ...],
    law_type: str,
    min_gate: float,
    temperature: float,
    beta_min: float,
    beta_max: float,
) -> tuple[float, float, float, float]:
    if law_type == "learned_residual" or not output_indices:
        return 0.0, 0.0, 1.0, 0.0

    index = torch.tensor(output_indices, dtype=torch.long, device=raw_prior.device)
    prior = raw_prior.index_select(dim=-1, index=index).float()
    target = target_delta.index_select(dim=-1, index=index).float()
    if context is None:
        weights = torch.ones(prior.shape[0], dtype=prior.dtype, device=prior.device)
        signed_prior = prior
    else:
        context = context.float()
        weights = context.abs()
        signed_prior = prior * context.unsqueeze(-1)

    if float(weights.max().detach().cpu()) <= 1e-8:
        return 0.0, 0.0, 1.0, 0.0
    weights = weights / weights.mean().clamp_min(1e-6)
    prior_centered = signed_prior - _weighted_mean(signed_prior, weights)
    target_centered = target - _weighted_mean(target, weights)

    weighted_prior = prior_centered * weights.unsqueeze(-1).sqrt()
    weighted_target = target_centered * weights.unsqueeze(-1).sqrt()
    prior_energy = weighted_prior.pow(2).sum()
    target_energy = weighted_target.pow(2).sum()
    if float(prior_energy.detach().cpu()) <= 1e-10 or float(target_energy.detach().cpu()) <= 1e-10:
        return 0.0, 0.0, 1.0, 0.0

    dot = (weighted_prior * weighted_target).sum()
    scale = dot / prior_energy.clamp_min(1e-10)
    prediction = scale * weighted_prior
    mse_prior = (prediction - weighted_target).pow(2).mean()
    mse_zero = weighted_target.pow(2).mean().clamp_min(1e-10)
    improvement = (1.0 - mse_prior / mse_zero).clamp(min=0.0, max=1.0)
    cosine = (dot / (prior_energy.sqrt() * target_energy.sqrt()).clamp_min(1e-10)).clamp(
        min=0.0,
        max=1.0,
    )
    raw_score = torch.sqrt(improvement * cosine).clamp(min=0.0, max=1.0)
    temp = max(1e-6, float(temperature))
    score = raw_score / (raw_score + temp)
    score_value = float(score.detach().cpu())
    min_gate = float(max(0.0, min(1.0, min_gate)))
    gate = min_gate + (1.0 - min_gate) * score_value
    beta = float(scale.abs().clamp(min=float(beta_min), max=float(beta_max)).detach().cpu())
    return gate, score_value, beta, float(raw_score.detach().cpu())


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    normalizer = weights.sum().clamp_min(1e-8)
    return (values * weights.unsqueeze(-1)).sum(dim=0, keepdim=True) / normalizer


def residual_scale_for_epoch(epoch: int, epochs: int, warmup_fraction: float) -> float:
    if warmup_fraction <= 0.0 or epochs <= 1:
        return 1.0
    warmup_epochs = max(1, int(round(float(epochs) * float(warmup_fraction))))
    return min(1.0, max(0.0, float(epoch + 1) / float(warmup_epochs)))


def _rollout_loss_for_batch(
    model: DUCWorldModel,
    transitions: MuJoCoTransitions,
    batch: DUCBatch,
    horizon: int,
    control_weights: torch.Tensor,
    teacher_force_context: bool,
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
        _history_for_indices(
            transitions.states,
            index_np,
            config_history_length(batch),
            dones=transitions.dones,
        ),
        dtype=torch.float32,
        device=device,
    )
    history_actions = torch.tensor(
        _history_for_indices(
            transitions.actions,
            index_np,
            config_history_length(batch),
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
        if teacher_force_context and transitions.contexts is not None:
            context = torch.tensor(transitions.contexts[step_indices], dtype=torch.float32, device=device)
        output = model(
            current,
            actions,
            history_states,
            history_actions,
            context=context,
            sample_context=False,
        )
        weights = control_weights.unsqueeze(0).expand_as(targets)
        total = total + weighted_mse(output.mean, targets, weights)
        current = output.mean
        history_states = torch.cat([history_states[:, 1:], current.unsqueeze(1)], dim=1)
        history_actions = torch.cat([history_actions[:, 1:], actions.unsqueeze(1)], dim=1)
    return total / float(horizon)


def config_history_length(batch: DUCBatch) -> int:
    return int(batch.history_states.shape[1])


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
