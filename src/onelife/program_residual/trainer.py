from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .core import ModelOutput, TransitionBatch
from .model import ProgramResidualWorldModel


@dataclass(frozen=True)
class ProgramResidualTrainerConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    residual_l2_weight: float = 1e-3
    max_grad_norm: float | None = 1.0


@dataclass(frozen=True)
class TrainingMetrics:
    loss: float
    prediction_loss: float
    residual_l2: float
    mean_unknown_fraction: float
    mean_symbolic_gate: float | None = None


def compute_program_residual_loss(
    output: ModelOutput,
    target_next_states: torch.Tensor,
    residual_l2_weight: float = 1e-3,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prediction_loss = F.mse_loss(output.prediction, target_next_states)
    residual_l2 = torch.mean(output.applied_residual.square())
    loss = prediction_loss + residual_l2_weight * residual_l2
    metrics = {
        "prediction_loss": prediction_loss.detach(),
        "residual_l2": residual_l2.detach(),
        "mean_unknown_fraction": output.unknown_mask.float().mean().detach(),
    }
    symbolic_gate = getattr(output, "symbolic_gate", None)
    if symbolic_gate is not None:
        metrics["mean_symbolic_gate"] = symbolic_gate.float().mean().detach()
    return loss, metrics


def make_optimizer(
    model: ProgramResidualWorldModel,
    config: ProgramResidualTrainerConfig,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def train_step(
    model: ProgramResidualWorldModel,
    optimizer: torch.optim.Optimizer,
    batch: TransitionBatch,
    config: ProgramResidualTrainerConfig,
) -> TrainingMetrics:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(batch.states, batch.actions)
    loss, tensors = compute_program_residual_loss(
        output=output,
        target_next_states=batch.next_states,
        residual_l2_weight=config.residual_l2_weight,
    )
    loss.backward()
    if config.max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
    optimizer.step()
    return TrainingMetrics(
        loss=float(loss.detach().cpu()),
        prediction_loss=float(tensors["prediction_loss"].cpu()),
        residual_l2=float(tensors["residual_l2"].cpu()),
        mean_unknown_fraction=float(tensors["mean_unknown_fraction"].cpu()),
        mean_symbolic_gate=float(tensors["mean_symbolic_gate"].cpu())
        if "mean_symbolic_gate" in tensors
        else None,
    )


def fit_supervised(
    model: ProgramResidualWorldModel,
    batches: Iterable[TransitionBatch],
    config: ProgramResidualTrainerConfig | None = None,
    num_epochs: int = 1,
) -> list[TrainingMetrics]:
    cfg = config or ProgramResidualTrainerConfig()
    optimizer = make_optimizer(model, cfg)
    history: list[TrainingMetrics] = []
    for _ in range(num_epochs):
        for batch in batches:
            history.append(train_step(model, optimizer, batch, cfg))
    return history
