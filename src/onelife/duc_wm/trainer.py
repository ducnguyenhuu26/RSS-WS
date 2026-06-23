from __future__ import annotations

from dataclasses import dataclass

import torch

from onelife.mujoco_dataset import MuJoCoTransitions

from .data import DUCBatch, align_contexts_to_templates, iter_duc_batches
from .losses import DUCLossConfig, compute_duc_loss, weighted_mse
from .metrics import default_control_weights
from .model import DUCWorldModel


@dataclass(frozen=True)
class DUCTrainerConfig:
    epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 1e-3
    history_length: int = 4
    beta_kl: float = 1e-3
    context_weight: float = 1.0
    control_weight: float = 0.0
    rollout_weight: float = 0.0
    rollout_horizon: int = 1
    orth_weight: float = 0.0
    sparse_weight: float = 0.0
    teacher_force_context: bool = True
    seed: int = 0


def fit_duc_world_model(
    model: DUCWorldModel,
    transitions: MuJoCoTransitions,
    config: DUCTrainerConfig,
    device: torch.device | str,
) -> list[dict[str, float]]:
    model.to(device)
    transitions = align_contexts_to_templates(transitions, model.config.templates)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loss_config = DUCLossConfig(
        beta_kl=config.beta_kl,
        context_weight=config.context_weight,
        control_weight=config.control_weight,
        orth_weight=config.orth_weight,
        sparse_weight=config.sparse_weight,
    )
    control_weights_np = default_control_weights(transitions.state_dim, model.config.templates)
    control_weights = torch.tensor(control_weights_np, dtype=torch.float32, device=device)
    history: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        totals: list[float] = []
        nlls: list[float] = []
        kls: list[float] = []
        ctxs: list[float] = []
        rolls: list[float] = []
        for batch in iter_duc_batches(
            transitions,
            batch_size=config.batch_size,
            history_length=config.history_length,
            shuffle=True,
            seed=config.seed + epoch,
            device=device,
        ):
            optimizer.zero_grad(set_to_none=True)
            context = batch.contexts if config.teacher_force_context else None
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
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            totals.append(float(total.detach().cpu()))
            nlls.append(float(loss.nll.cpu()))
            kls.append(float(loss.kl.cpu()))
            ctxs.append(float(loss.context.cpu()))
            rolls.append(float(rollout.detach().cpu()))
        history.append(
            {
                "epoch": float(epoch + 1),
                "loss": sum(totals) / max(1, len(totals)),
                "nll": sum(nlls) / max(1, len(nlls)),
                "kl": sum(kls) / max(1, len(kls)),
                "context": sum(ctxs) / max(1, len(ctxs)),
                "rollout": sum(rolls) / max(1, len(rolls)),
            }
        )
    return history


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
    total = current.new_zeros(())
    for offset in range(horizon):
        step_indices = index_np + offset
        actions = torch.tensor(transitions.actions[step_indices], dtype=torch.float32, device=device)
        targets = torch.tensor(transitions.next_states[step_indices], dtype=torch.float32, device=device)
        context = None
        if teacher_force_context and transitions.contexts is not None:
            context = torch.tensor(transitions.contexts[step_indices], dtype=torch.float32, device=device)
        output = model(current, actions, context=context, sample_context=False)
        weights = control_weights.unsqueeze(0).expand_as(targets)
        total = total + weighted_mse(output.mean, targets, weights)
        current = output.mean
    return total / float(horizon)
