from __future__ import annotations

from dataclasses import dataclass

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
    trust_region_weight: float = 0.5
    trust_region_delta_min: float = 0.15
    trust_region_delta_range: float = 0.75
    prior_beta_weight: float = 1e-4
    residual_warmup_fraction: float = 0.5
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
