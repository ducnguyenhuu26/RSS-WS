from __future__ import annotations

from dataclasses import dataclass

import torch

from onelife.mujoco_dataset import MuJoCoTransitions

from .data import align_contexts_to_templates, iter_duc_batches
from .losses import DUCLossConfig, compute_duc_loss
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
            loss.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            totals.append(float(loss.total.detach().cpu()))
            nlls.append(float(loss.nll.cpu()))
            kls.append(float(loss.kl.cpu()))
            ctxs.append(float(loss.context.cpu()))
        history.append(
            {
                "epoch": float(epoch + 1),
                "loss": sum(totals) / max(1, len(totals)),
                "nll": sum(nlls) / max(1, len(nlls)),
                "kl": sum(kls) / max(1, len(kls)),
                "context": sum(ctxs) / max(1, len(ctxs)),
            }
        )
    return history
