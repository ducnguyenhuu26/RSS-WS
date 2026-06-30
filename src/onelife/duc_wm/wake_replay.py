from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from .mujoco_ext import (
    apply_action_context,
    apply_parameter_context,
    apply_transition_context,
    sample_context,
)
from .planner import CEMPlannerConfig, rollout_action_sequences
from .planning_eval import PlannerCoverageStats, align_raw_context_to_templates
from .reward_model import RewardModel
from .templates import MechanismTemplate


@dataclass(frozen=True)
class WakeReplayConfig:
    enabled: bool = False
    episodes: int = 3
    max_steps: int = 90
    horizon: int = 15
    candidates: int = 256
    elites: int = 32
    iterations: int = 3
    uncertainty_weight: float = 0.0
    model_bonus_weight: float = 0.0
    certified_risk_weight: float = 0.0
    risk_delta: float = 0.10
    posterior_trust: float = 0.15
    posterior_temperature: float = 1.0


@dataclass(frozen=True)
class WakeSegment:
    predicted_return: float
    true_return: float
    risk_sum: float
    bonus_sum: float
    uncertainty_sum: float
    ood_mean: float
    alpha_mean: np.ndarray
    steps: int

    @property
    def abs_gap(self) -> float:
        return abs(self.predicted_return - self.true_return)

    @property
    def optimism_gap(self) -> float:
        return max(0.0, self.predicted_return - self.true_return)


@torch.no_grad()
def calibrate_wake_replay(
    model: nn.Module,
    reward_model: RewardModel,
    env_id: str,
    variants: list[str],
    seed: int,
    action_smoothing: float,
    history_length: int,
    config: WakeReplayConfig,
    device: torch.device | str,
    use_oracle_context: bool = False,
    context_templates: tuple[MechanismTemplate, ...] = (),
    coverage_stats: PlannerCoverageStats | None = None,
) -> dict[str, float]:
    if not config.enabled:
        return {}
    if config.episodes <= 0 or config.max_steps <= 0 or config.horizon <= 0:
        return {}
    if not variants:
        raise ValueError("wake replay variants must be non-empty")

    was_training = model.training
    reward_was_training = reward_model.training
    previous_planning_mode = getattr(model, "planning_mode", None)
    model.eval()
    reward_model.eval()
    if hasattr(model, "set_planning_mode"):
        model.set_planning_mode(True)
    rng = np.random.default_rng(seed)
    segments: list[WakeSegment] = []
    try:
        for episode in range(config.episodes):
            variant = variants[episode % len(variants)]
            context = sample_context(variant, rng)
            segments.extend(
                _collect_wake_episode(
                    model=model,
                    reward_model=reward_model,
                    env_id=env_id,
                    context=context,
                    seed=seed + 10_003 * episode,
                    action_smoothing=action_smoothing,
                    history_length=history_length,
                    config=config,
                    device=device,
                    use_oracle_context=use_oracle_context,
                    context_templates=context_templates,
                    coverage_stats=coverage_stats,
                )
            )
    finally:
        if hasattr(model, "set_planning_mode") and previous_planning_mode is not None:
            model.set_planning_mode(bool(previous_planning_mode))
        if was_training:
            model.train()
        if reward_was_training:
            reward_model.train()

    if not segments:
        return {"wake_segments": 0.0, "wake_steps": 0.0}

    metrics = _calibrate_from_segments(
        model=model,
        segments=segments,
        delta=config.risk_delta,
        posterior_trust=config.posterior_trust,
        posterior_temperature=config.posterior_temperature,
        device=device,
    )
    return metrics


def _collect_wake_episode(
    model: nn.Module,
    reward_model: RewardModel,
    env_id: str,
    context: np.ndarray,
    seed: int,
    action_smoothing: float,
    history_length: int,
    config: WakeReplayConfig,
    device: torch.device | str,
    use_oracle_context: bool,
    context_templates: tuple[MechanismTemplate, ...],
    coverage_stats: PlannerCoverageStats | None,
) -> list[WakeSegment]:
    env = gym.make(env_id)
    segments: list[WakeSegment] = []
    try:
        apply_parameter_context(env, context)
        obs, _ = env.reset(seed=seed)
        obs = np.asarray(obs, dtype=np.float32)
        action_low = np.asarray(env.action_space.low, dtype=np.float32)
        action_high = np.asarray(env.action_space.high, dtype=np.float32)
        previous_action = np.zeros_like(action_low, dtype=np.float32)
        history_states = np.repeat(obs[None, :], history_length, axis=0).astype(np.float32)
        history_actions = np.repeat(previous_action[None, :], history_length, axis=0).astype(np.float32)
        episode_rng = np.random.default_rng(seed + 1)
        model_context = (
            align_raw_context_to_templates(context, context_templates)
            if use_oracle_context
            else None
        )
        elapsed = 0
        while elapsed < config.max_steps:
            horizon = min(config.horizon, config.max_steps - elapsed)
            if horizon <= 0:
                break
            sequence, predicted = _plan_action_sequence(
                model=model,
                reward_model=reward_model,
                state=obs,
                action_low=action_low,
                action_high=action_high,
                history_states=history_states,
                history_actions=history_actions,
                model_context=model_context,
                config=config,
                horizon=horizon,
                device=device,
            )
            true_return = 0.0
            steps = 0
            ood_values: list[float] = []
            done = False
            for action in sequence:
                action = np.clip(action.astype(np.float32), action_low, action_high)
                if coverage_stats is not None:
                    ood_values.append(coverage_stats.distance(obs, action))
                effective_action = apply_action_context(action, previous_action, context, episode_rng)
                env_next, reward, terminated, truncated, _ = env.step(effective_action)
                env_next = np.asarray(env_next, dtype=np.float32)
                obs = apply_transition_context(
                    state=obs,
                    next_state=env_next,
                    context=context,
                    rng=episode_rng,
                )
                true_return += float(reward)
                previous_action = action
                history_states = np.concatenate([history_states[1:], obs[None, :]], axis=0)
                history_actions = np.concatenate([history_actions[1:], action[None, :]], axis=0)
                steps += 1
                elapsed += 1
                done = bool(terminated or truncated)
                if done or elapsed >= config.max_steps:
                    break
            if steps <= 0:
                break
            segments.append(
                WakeSegment(
                    predicted_return=predicted["predicted_return"],
                    true_return=true_return,
                    risk_sum=predicted["risk_sum"],
                    bonus_sum=predicted["bonus_sum"],
                    uncertainty_sum=predicted["uncertainty_sum"],
                    ood_mean=float(np.mean(ood_values)) if ood_values else 0.0,
                    alpha_mean=predicted["alpha_mean"],
                    steps=steps,
                )
            )
            if done:
                obs, _ = env.reset(seed=seed + elapsed + 1)
                obs = np.asarray(obs, dtype=np.float32)
                previous_action = np.zeros_like(action_low, dtype=np.float32)
                history_states = np.repeat(obs[None, :], history_length, axis=0).astype(np.float32)
                history_actions = np.repeat(previous_action[None, :], history_length, axis=0).astype(np.float32)
    finally:
        env.close()
    return segments


def _plan_action_sequence(
    model: nn.Module,
    reward_model: RewardModel,
    state: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
    history_states: np.ndarray,
    history_actions: np.ndarray,
    model_context: np.ndarray | None,
    config: WakeReplayConfig,
    horizon: int,
    device: torch.device | str,
) -> tuple[np.ndarray, dict[str, float | np.ndarray]]:
    action_dim = int(action_low.shape[0])
    mean = torch.zeros(horizon, action_dim, device=device)
    std = torch.ones_like(mean)
    low = torch.tensor(action_low, dtype=torch.float32, device=device)
    high = torch.tensor(action_high, dtype=torch.float32, device=device)
    initial_state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    history_state_tensor = torch.tensor(history_states, dtype=torch.float32, device=device).unsqueeze(0)
    history_action_tensor = torch.tensor(history_actions, dtype=torch.float32, device=device).unsqueeze(0)
    context_tensor = None
    if model_context is not None:
        context_tensor = torch.tensor(model_context, dtype=torch.float32, device=device).unsqueeze(0)
    planner_config = CEMPlannerConfig(
        horizon=horizon,
        candidates=config.candidates,
        elites=config.elites,
        iterations=config.iterations,
        uncertainty_weight=config.uncertainty_weight,
        model_bonus_weight=config.model_bonus_weight,
        certified_risk_weight=config.certified_risk_weight,
    )
    for _ in range(config.iterations):
        samples = mean + std * torch.randn(config.candidates, horizon, action_dim, device=device)
        samples = torch.max(torch.min(samples, high), low)
        scores = rollout_action_sequences(
            model=model,
            initial_state=initial_state,
            action_sequences=samples,
            reward_fn=lambda states, actions, next_states: reward_model(states, actions, next_states),
            config=planner_config,
            history_states=history_state_tensor,
            history_actions=history_action_tensor,
            model_context=context_tensor,
        )
        elite_indices = scores.topk(k=min(config.elites, config.candidates)).indices
        elites = samples.index_select(dim=0, index=elite_indices)
        mean = elites.mean(dim=0)
        std = elites.std(dim=0).clamp_min(1e-3)
    stats = _rollout_sequence_stats(
        model=model,
        reward_model=reward_model,
        initial_state=initial_state,
        action_sequence=mean,
        history_states=history_state_tensor,
        history_actions=history_action_tensor,
        model_context=context_tensor,
    )
    return mean.detach().cpu().numpy().astype(np.float32), stats


def _rollout_sequence_stats(
    model: nn.Module,
    reward_model: RewardModel,
    initial_state: torch.Tensor,
    action_sequence: torch.Tensor,
    history_states: torch.Tensor,
    history_actions: torch.Tensor,
    model_context: torch.Tensor | None,
) -> dict[str, float | np.ndarray]:
    state = initial_state
    history_states = history_states.clone()
    history_actions = history_actions.clone()
    if model_context is not None:
        model_context = model_context.clone()
    use_belief_state = hasattr(model, "initial_belief_state")
    belief_state = None
    predicted_return = state.new_zeros(())
    risk_sum = state.new_zeros(())
    bonus_sum = state.new_zeros(())
    uncertainty_sum = state.new_zeros(())
    alpha_values: list[torch.Tensor] = []
    for step in range(action_sequence.shape[0]):
        action = action_sequence[step].unsqueeze(0)
        if use_belief_state:
            if belief_state is None:
                belief_state = model.initial_belief_state(
                    state,
                    action,
                    history_states,
                    history_actions,
                )
            output = model(
                state,
                action,
                history_states,
                history_actions,
                context=model_context,
                sample_context=False,
                belief_state=belief_state,
            )
            belief_state = getattr(output, "belief_next", belief_state)
        else:
            output = model(
                state,
                action,
                history_states,
                history_actions,
                context=model_context,
                sample_context=False,
            )
        reward = reward_model(state, action, output.mean).mean()
        predicted_return = predicted_return + reward
        risk = getattr(output, "certified_risk", torch.zeros_like(reward)).mean()
        bonus = getattr(output, "planning_bonus", torch.zeros_like(reward)).mean()
        uncertainty = torch.exp(output.logvar).mean()
        risk_sum = risk_sum + risk
        bonus_sum = bonus_sum + bonus
        uncertainty_sum = uncertainty_sum + uncertainty
        alpha = getattr(output, "alpha_ctrl_mean", getattr(output, "alpha_mean", None))
        if alpha is not None:
            alpha_values.append(alpha.detach().mean(dim=0))
        state = output.mean
        history_states = torch.cat([history_states[:, 1:], state.unsqueeze(1)], dim=1)
        history_actions = torch.cat([history_actions[:, 1:], action.unsqueeze(1)], dim=1)
    if alpha_values:
        alpha_mean = torch.stack(alpha_values, dim=0).mean(dim=0).detach().cpu().numpy()
    else:
        alpha_mean = np.zeros(1, dtype=np.float32)
    return {
        "predicted_return": float(predicted_return.detach().cpu()),
        "risk_sum": float(risk_sum.detach().cpu()),
        "bonus_sum": float(bonus_sum.detach().cpu()),
        "uncertainty_sum": float(uncertainty_sum.detach().cpu()),
        "alpha_mean": alpha_mean.astype(np.float32),
    }


def _calibrate_from_segments(
    model: nn.Module,
    segments: list[WakeSegment],
    delta: float,
    posterior_trust: float,
    posterior_temperature: float,
    device: torch.device | str,
) -> dict[str, float]:
    predicted = np.asarray([item.predicted_return for item in segments], dtype=np.float64)
    true = np.asarray([item.true_return for item in segments], dtype=np.float64)
    risks = np.asarray([max(1e-6, item.risk_sum) for item in segments], dtype=np.float64)
    bonuses = np.asarray([item.bonus_sum for item in segments], dtype=np.float64)
    uncertainties = np.asarray([item.uncertainty_sum for item in segments], dtype=np.float64)
    oods = np.asarray([item.ood_mean for item in segments], dtype=np.float64)
    steps = np.asarray([item.steps for item in segments], dtype=np.float64)
    abs_gaps = np.abs(predicted - true)
    optimism = np.maximum(0.0, predicted - true)
    quantile = float(np.clip(1.0 - float(delta), 0.0, 1.0))
    ratios = optimism / np.maximum(risks, 1e-6)
    wake_scale = float(np.quantile(ratios, quantile))
    previous_scale = _current_risk_scale(model)
    scale = max(previous_scale, wake_scale)
    coverage = float(np.mean(optimism <= scale * risks + 1e-8))
    if hasattr(model, "set_certified_risk_calibration"):
        model.set_certified_risk_calibration(
            scale=scale,
            coverage=coverage,
            gap_mean=float(np.mean(abs_gaps)),
        )
    if hasattr(model, "update_law_posterior"):
        alphas = _stack_alpha_means(segments)
        if alphas.size > 0 and alphas.shape[1] > 0:
            centered_gap = optimism - float(np.mean(optimism))
            gap_scale = float(np.std(centered_gap) + 1e-6)
            evidence_np = -np.mean(alphas * (centered_gap / gap_scale)[:, None], axis=0)
            evidence = torch.tensor(evidence_np, dtype=torch.float32, device=device)
            model.update_law_posterior(
                evidence=evidence,
                trust=posterior_trust,
                temperature=posterior_temperature,
            )
        else:
            evidence_np = np.asarray([], dtype=np.float32)
    else:
        evidence_np = np.asarray([], dtype=np.float32)

    lower_bound = predicted - scale * risks
    return {
        "wake_segments": float(len(segments)),
        "wake_steps": float(np.sum(steps)),
        "wake_predicted_return_mean": float(np.mean(predicted)),
        "wake_true_return_mean": float(np.mean(true)),
        "wake_abs_gap_mean": float(np.mean(abs_gaps)),
        "wake_optimism_gap_mean": float(np.mean(optimism)),
        "wake_risk_sum_mean": float(np.mean(risks)),
        "wake_risk_scale_raw": wake_scale,
        "wake_risk_scale": scale,
        "wake_risk_coverage": coverage,
        "wake_lcb_return_mean": float(np.mean(lower_bound)),
        "wake_bonus_sum_mean": float(np.mean(bonuses)),
        "wake_uncertainty_sum_mean": float(np.mean(uncertainties)),
        "wake_ood_mean": float(np.mean(oods)),
        "wake_selection_gap_cov": _covariance(predicted, optimism),
        "wake_risk_gap_cov": _covariance(risks, optimism),
        "wake_bonus_gap_cov": _covariance(bonuses, optimism),
        "wake_posterior_evidence_mean": float(np.mean(evidence_np)) if evidence_np.size else 0.0,
        "wake_posterior_evidence_std": float(np.std(evidence_np)) if evidence_np.size else 0.0,
    }


def _stack_alpha_means(segments: list[WakeSegment]) -> np.ndarray:
    widths = [item.alpha_mean.shape[0] for item in segments if item.alpha_mean.size > 0]
    if not widths:
        return np.empty((0, 0), dtype=np.float32)
    width = max(widths)
    rows = []
    for item in segments:
        row = np.zeros(width, dtype=np.float32)
        count = min(width, item.alpha_mean.shape[0])
        row[:count] = item.alpha_mean[:count]
        rows.append(row)
    return np.stack(rows, axis=0)


def _covariance(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or right.size < 2:
        return 0.0
    return float(np.mean((left - left.mean()) * (right - right.mean())))


def _current_risk_scale(model: nn.Module) -> float:
    value = getattr(model, "certified_risk_scale", 1.0)
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    try:
        return float(value)
    except (TypeError, ValueError):
        return 1.0
