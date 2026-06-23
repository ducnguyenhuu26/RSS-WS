from __future__ import annotations

import numpy as np
import torch

from onelife.mujoco_dataset import MuJoCoTransitions

from .data import align_contexts_to_templates, iter_duc_batches
from .model import DUCWorldModel
from .templates import MechanismTemplate


def default_control_weights(
    state_dim: int,
    templates: tuple[MechanismTemplate, ...],
) -> np.ndarray:
    weights = np.ones(state_dim, dtype=np.float32)
    for template in templates:
        boost = 3.0 if template.name in {"actuation", "wind", "friction", "sticky"} else 1.5
        for index in template.output_indices:
            weights[index] = max(weights[index], boost)
    return weights


def r2_score(prediction: np.ndarray, target: np.ndarray, weights: np.ndarray | None = None) -> float:
    error = target - prediction
    centered = target - target.mean(axis=0, keepdims=True)
    if weights is not None:
        error = error * np.sqrt(weights[None, :])
        centered = centered * np.sqrt(weights[None, :])
    numerator = float(np.sum(error**2))
    denominator = float(np.sum(centered**2))
    if denominator <= 1e-12:
        return 0.0
    return 1.0 - numerator / denominator


@torch.no_grad()
def evaluate_duc_model(
    model: DUCWorldModel,
    transitions: MuJoCoTransitions,
    device: torch.device | str,
    batch_size: int = 512,
    history_length: int = 4,
) -> dict[str, float]:
    model.eval()
    transitions = align_contexts_to_templates(transitions, model.config.templates)
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    context_errors: list[float] = []
    for batch in iter_duc_batches(
        transitions,
        batch_size=batch_size,
        history_length=history_length,
        shuffle=False,
        device=device,
    ):
        output = model(
            batch.states,
            batch.actions,
            batch.history_states,
            batch.history_actions,
            context=None,
            sample_context=False,
        )
        predictions.append(output.mean.cpu().numpy())
        targets.append(batch.next_states.cpu().numpy())
        if batch.contexts is not None:
            context_errors.append(
                float((output.alpha_mean - batch.contexts).pow(2).mean().cpu())
            )
    pred = np.concatenate(predictions, axis=0)
    target = np.concatenate(targets, axis=0)
    weights = default_control_weights(transitions.state_dim, model.config.templates)
    metrics = {
        "r2_at_1": r2_score(pred, target),
        "duc_r2_at_1": r2_score(pred, target, weights=weights),
        "mse": float(np.mean((pred - target) ** 2)),
    }
    if context_errors:
        metrics["context_mse"] = float(sum(context_errors) / len(context_errors))
    return metrics
