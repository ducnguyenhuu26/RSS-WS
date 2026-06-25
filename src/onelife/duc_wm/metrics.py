from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

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
        if template.name == "unknown" or template.timescale == "unknown":
            continue
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


def attribution_accuracy(
    alpha: np.ndarray,
    contexts: np.ndarray,
    templates: tuple[MechanismTemplate, ...],
    threshold: float = 1e-6,
) -> float | None:
    candidate_indices = [
        index
        for index, template in enumerate(templates)
        if template.name not in {"actuation", "unknown"}
        and template.timescale != "unknown"
    ]
    if not candidate_indices:
        return None
    true_signal = np.abs(contexts[:, candidate_indices])
    pred_signal = np.abs(alpha[:, candidate_indices])
    active = true_signal.max(axis=1) > threshold
    if not bool(active.any()):
        return None
    true_top = true_signal[active].argmax(axis=1)
    pred_top = pred_signal[active].argmax(axis=1)
    return float(np.mean(true_top == pred_top))


def attribution_recall_at_k(
    alpha: np.ndarray,
    contexts: np.ndarray,
    templates: tuple[MechanismTemplate, ...],
    k: int = 2,
    threshold: float = 1e-6,
) -> float | None:
    candidate_indices = [
        index
        for index, template in enumerate(templates)
        if template.name not in {"actuation", "unknown"}
        and template.timescale != "unknown"
    ]
    if not candidate_indices:
        return None
    true_signal = np.abs(contexts[:, candidate_indices])
    pred_signal = np.abs(alpha[:, candidate_indices])
    true_active = true_signal > threshold
    active_rows = true_active.any(axis=1)
    if not bool(active_rows.any()):
        return None
    recalls = []
    top_k = min(k, len(candidate_indices))
    pred_top = np.argsort(-pred_signal[active_rows], axis=1)[:, :top_k]
    true_active = true_active[active_rows]
    for row, pred_indices in enumerate(pred_top):
        true_indices = set(np.flatnonzero(true_active[row]))
        if not true_indices:
            continue
        pred_set = set(int(index) for index in pred_indices)
        recalls.append(len(true_indices.intersection(pred_set)) / len(true_indices))
    if not recalls:
        return None
    return float(np.mean(recalls))


def strength_spearman(
    alpha: np.ndarray,
    contexts: np.ndarray,
    templates: tuple[MechanismTemplate, ...],
    threshold: float = 1e-8,
) -> float | None:
    candidate_indices = [
        index
        for index, template in enumerate(templates)
        if template.name not in {"actuation", "unknown"}
        and template.timescale != "unknown"
    ]
    scores = []
    for index in candidate_indices:
        truth = contexts[:, index]
        pred = alpha[:, index]
        if float(np.std(truth)) <= threshold or float(np.std(pred)) <= threshold:
            continue
        truth_rank = _rankdata(truth)
        pred_rank = _rankdata(pred)
        corr = np.corrcoef(truth_rank, pred_rank)[0, 1]
        if np.isfinite(corr):
            scores.append(float(corr))
    if not scores:
        return None
    return float(np.mean(scores))


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(len(values), dtype=np.float32)
    return ranks


@torch.no_grad()
def evaluate_world_model(
    model: nn.Module,
    transitions: MuJoCoTransitions,
    device: torch.device | str,
    control_templates: tuple[MechanismTemplate, ...],
    batch_size: int = 512,
    history_length: int = 4,
    rollout_horizon: int = 5,
    use_oracle_context: bool = False,
) -> dict[str, float]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    nlls: list[float] = []
    for batch in iter_duc_batches(
        transitions,
        batch_size=batch_size,
        history_length=history_length,
        shuffle=False,
        device=device,
    ):
        context = batch.contexts if use_oracle_context else None
        output = model(
            batch.states,
            batch.actions,
            batch.history_states,
            batch.history_actions,
            context=context,
            sample_context=False,
        )
        predictions.append(output.mean.cpu().numpy())
        targets.append(batch.next_states.cpu().numpy())
        if hasattr(model, "nll"):
            nlls.append(float(model.nll(output, batch.next_states).cpu()))
    pred = np.concatenate(predictions, axis=0)
    target = np.concatenate(targets, axis=0)
    weights = default_control_weights(transitions.state_dim, control_templates)
    metrics = {
        "r2_at_1": r2_score(pred, target),
        "duc_r2_at_1": r2_score(pred, target, weights=weights),
        "mse": float(np.mean((pred - target) ** 2)),
    }
    if nlls:
        metrics["nll"] = float(sum(nlls) / len(nlls))
    rollout = rollout_predictions(
        model=model,
        transitions=transitions,
        device=device,
        batch_size=batch_size,
        history_length=history_length,
        horizon=rollout_horizon,
        use_oracle_context=use_oracle_context,
    )
    if rollout is not None:
        rollout_pred, rollout_target = rollout
        metrics[f"r2_at_{rollout_horizon}"] = r2_score(rollout_pred, rollout_target)
        metrics[f"duc_r2_at_{rollout_horizon}"] = r2_score(
            rollout_pred,
            rollout_target,
            weights=weights,
        )
        metrics[f"mse_at_{rollout_horizon}"] = float(
            np.mean((rollout_pred - rollout_target) ** 2)
        )
    return metrics


@torch.no_grad()
def evaluate_duc_model(
    model: DUCWorldModel,
    transitions: MuJoCoTransitions,
    device: torch.device | str,
    batch_size: int = 512,
    history_length: int = 4,
    rollout_horizon: int = 5,
) -> dict[str, float]:
    model.eval()
    transitions = align_contexts_to_templates(transitions, model.config.templates)
    metrics = evaluate_world_model(
        model=model,
        transitions=transitions,
        device=device,
        control_templates=model.config.templates,
        batch_size=batch_size,
        history_length=history_length,
        rollout_horizon=rollout_horizon,
    )
    alphas: list[np.ndarray] = []
    context_errors: list[float] = []
    prior_norms: list[float] = []
    residual_norms: list[float] = []
    mechanism_norms: list[float] = []
    context_norms: list[float] = []
    proposed_norms: list[float] = []
    planning_delta_norms: list[float] = []
    mechanism_mixes: list[float] = []
    trust_violations: list[float] = []
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
        alphas.append(output.alpha_mean.cpu().numpy())
        prior_norms.append(float(output.prior_delta.norm(dim=-1).mean().cpu()))
        residual_norms.append(float(output.residual_delta.norm(dim=-1).mean().cpu()))
        mechanism_norms.append(float(output.mechanism_delta.norm(dim=-1).mean().cpu()))
        context_norms.append(float(output.context_delta.norm(dim=-1).mean().cpu()))
        proposed_norms.append(float(output.proposed_mechanism_delta.norm(dim=-1).mean().cpu()))
        planning_delta_norms.append(float(output.planning_delta.norm(dim=-1).mean().cpu()))
        mechanism_mixes.append(float(output.mechanism_mix.mean().cpu()))
        trust_violations.append(float(trust_region_violation(model, output).cpu()))
        if batch.contexts is not None:
            context_errors.append(
                float((output.alpha_mean - batch.contexts).pow(2).mean().cpu())
            )
    alpha = np.concatenate(alphas, axis=0)
    if prior_norms:
        prior_norm = float(sum(prior_norms) / len(prior_norms))
        residual_norm = float(sum(residual_norms) / len(residual_norms))
        mechanism_norm = float(sum(mechanism_norms) / len(mechanism_norms))
        context_norm = float(sum(context_norms) / len(context_norms))
        proposed_norm = float(sum(proposed_norms) / len(proposed_norms))
        planning_delta_norm = float(sum(planning_delta_norms) / len(planning_delta_norms))
        metrics["prior_delta_norm"] = prior_norm
        metrics["residual_delta_norm"] = residual_norm
        metrics["mechanism_delta_norm"] = mechanism_norm
        metrics["context_delta_norm"] = context_norm
        metrics["proposed_mechanism_delta_norm"] = proposed_norm
        metrics["planning_delta_norm"] = planning_delta_norm
        metrics["prior_to_total_delta_ratio"] = prior_norm / max(1e-8, mechanism_norm)
        metrics["residual_to_total_delta_ratio"] = residual_norm / max(1e-8, mechanism_norm)
        metrics["prior_to_proposed_delta_ratio"] = prior_norm / max(1e-8, proposed_norm)
        metrics["residual_to_proposed_delta_ratio"] = residual_norm / max(1e-8, proposed_norm)
        metrics["proposed_to_final_delta_ratio"] = proposed_norm / max(1e-8, mechanism_norm)
        metrics["mechanism_mix_mean"] = float(sum(mechanism_mixes) / len(mechanism_mixes))
        metrics["trust_region_violation"] = float(sum(trust_violations) / len(trust_violations))
        beta = model.prior_beta.detach().cpu().numpy()
        metrics["prior_beta_mean"] = float(np.mean(beta))
        metrics["prior_beta_min"] = float(np.min(beta))
        metrics["prior_beta_max"] = float(np.max(beta))
        gate = model.prior_gate.detach().cpu().numpy()
        data_conf = model.data_confidence.detach().cpu().numpy()
        effective_conf = model.effective_prior_confidence.detach().cpu().numpy()
        metrics["prior_gate_mean"] = float(np.mean(gate))
        metrics["prior_gate_min"] = float(np.min(gate))
        metrics["prior_gate_max"] = float(np.max(gate))
        metrics["data_confidence_mean"] = float(np.mean(data_conf))
        metrics["data_confidence_min"] = float(np.min(data_conf))
        metrics["data_confidence_max"] = float(np.max(data_conf))
        metrics["effective_prior_confidence_mean"] = float(np.mean(effective_conf))
        reward_sensitivity = model.reward_sensitivity.detach().cpu().numpy()
        metrics["reward_sensitivity_mean"] = float(np.mean(reward_sensitivity))
        metrics["reward_sensitivity_max"] = float(np.max(reward_sensitivity))
        metrics["residual_scale"] = float(model._residual_scale.detach().cpu())
    if context_errors:
        metrics["context_mse"] = float(sum(context_errors) / len(context_errors))
    if transitions.contexts is not None:
        accuracy = attribution_accuracy(alpha, transitions.contexts, model.config.templates)
        if accuracy is not None:
            metrics["attribution_accuracy"] = accuracy
        recall = attribution_recall_at_k(alpha, transitions.contexts, model.config.templates, k=2)
        if recall is not None:
            metrics["attribution_recall_at_2"] = recall
        corr = strength_spearman(alpha, transitions.contexts, model.config.templates)
        if corr is not None:
            metrics["strength_spearman"] = corr
    if model.unknown_indices:
        metrics["unknown_alpha_abs"] = float(np.mean(np.abs(alpha[:, model.unknown_indices])))
    return metrics


def trust_region_violation(model: DUCWorldModel, output) -> torch.Tensor:
    confidences = model.effective_prior_confidence.to(output.residual_effects.device)
    prior_norm = output.prior_effects.pow(2).mean(dim=-1).add(1e-8).sqrt()
    residual_norm = output.residual_effects.pow(2).mean(dim=-1).add(1e-8).sqrt()
    delta = (
        float(model.config.trust_region_delta_min)
        + (1.0 - confidences) * float(model.config.trust_region_delta_range)
    )
    violation = torch.relu(residual_norm - delta.unsqueeze(0) * prior_norm)
    return (confidences.unsqueeze(0) * violation).mean()


@torch.no_grad()
def rollout_predictions(
    model: nn.Module,
    transitions: MuJoCoTransitions,
    device: torch.device | str,
    batch_size: int,
    history_length: int,
    horizon: int,
    use_oracle_context: bool = False,
) -> tuple[np.ndarray, np.ndarray] | None:
    if horizon <= 1 or transitions.num_steps < horizon:
        return None
    starts = np.arange(0, transitions.num_steps - horizon + 1)
    if transitions.dones is not None:
        keep = []
        for start in starts:
            done_window = transitions.dones[start : start + horizon - 1]
            if not bool(done_window.any()):
                keep.append(start)
        starts = np.asarray(keep, dtype=np.int64)
    if len(starts) == 0:
        return None

    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for begin in range(0, len(starts), batch_size):
        batch_starts = starts[begin : begin + batch_size]
        current = torch.tensor(
            transitions.states[batch_starts],
            dtype=torch.float32,
            device=device,
        )
        history_states = torch.tensor(
            _history_for_indices(
                transitions.states,
                batch_starts,
                history_length,
                dones=transitions.dones,
            ),
            dtype=torch.float32,
            device=device,
        )
        history_actions = torch.tensor(
            _history_for_indices(
                transitions.actions,
                batch_starts,
                history_length,
                dones=transitions.dones,
            ),
            dtype=torch.float32,
            device=device,
        )
        batch_predictions: list[np.ndarray] = []
        batch_targets: list[np.ndarray] = []
        for offset in range(horizon):
            step_indices = batch_starts + offset
            actions = torch.tensor(
                transitions.actions[step_indices],
                dtype=torch.float32,
                device=device,
            )
            context = None
            if use_oracle_context and transitions.contexts is not None:
                context = torch.tensor(
                    transitions.contexts[step_indices],
                    dtype=torch.float32,
                    device=device,
                )
            output = model(
                current,
                actions,
                history_states,
                history_actions,
                context=context,
                sample_context=False,
            )
            current = output.mean
            batch_predictions.append(current.cpu().numpy())
            batch_targets.append(transitions.next_states[step_indices])
            history_states = torch.cat([history_states[:, 1:], current.unsqueeze(1)], dim=1)
            history_actions = torch.cat([history_actions[:, 1:], actions.unsqueeze(1)], dim=1)
        predictions.append(np.stack(batch_predictions, axis=1).reshape(-1, transitions.state_dim))
        targets.append(np.stack(batch_targets, axis=1).reshape(-1, transitions.state_dim))
    return np.concatenate(predictions, axis=0), np.concatenate(targets, axis=0)


def _history_for_indices(
    values: np.ndarray,
    indices: np.ndarray,
    history_length: int,
    dones: np.ndarray | None = None,
) -> np.ndarray:
    history = np.zeros((len(indices), history_length, values.shape[1]), dtype=np.float32)
    for row, index in enumerate(indices):
        start = max(0, int(index) - history_length + 1)
        if dones is not None and int(index) > 0:
            done_positions = np.flatnonzero(dones[start:int(index)])
            if len(done_positions) > 0:
                start = start + int(done_positions[-1]) + 1
        window = values[start : int(index) + 1]
        if len(window) < history_length:
            pad = np.repeat(window[:1], history_length - len(window), axis=0)
            window = np.concatenate([pad, window], axis=0)
        history[row] = window[-history_length:]
    return history
