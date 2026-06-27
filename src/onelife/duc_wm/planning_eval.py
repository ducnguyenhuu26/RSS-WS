from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from onelife.mujoco_dataset import MuJoCoTransitions

from .mujoco_ext import (
    CONTEXT_NAMES,
    apply_action_context,
    apply_parameter_context,
    apply_transition_context,
    sample_context,
)
from .planner import CEMPlannerConfig, plan_cem_action
from .reward_model import RewardModel
from .templates import MechanismTemplate


@dataclass(frozen=True)
class PlanningEvalConfig:
    enabled: bool = False
    episodes: int = 3
    max_steps: int = 200
    horizon: int = 15
    candidates: int = 256
    elites: int = 32
    iterations: int = 3
    uncertainty_weight: float = 0.0
    model_bonus_weight: float = 0.0
    certified_risk_weight: float = 0.0


@dataclass(frozen=True)
class PlannerCoverageStats:
    mean: np.ndarray
    scale: np.ndarray
    train_p95: float

    def distance(self, state: np.ndarray, action: np.ndarray) -> float:
        vector = np.concatenate([state.astype(np.float32), action.astype(np.float32)])
        normalized = (vector - self.mean) / self.scale
        return float(np.sqrt(np.mean(np.square(normalized))))


def build_planner_coverage_stats(transitions: MuJoCoTransitions) -> PlannerCoverageStats:
    vectors = np.concatenate([transitions.states, transitions.actions], axis=1).astype(np.float32)
    mean = vectors.mean(axis=0)
    scale = vectors.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale).astype(np.float32)
    distances = np.sqrt(np.mean(np.square((vectors - mean) / scale), axis=1))
    return PlannerCoverageStats(
        mean=mean.astype(np.float32),
        scale=scale,
        train_p95=float(np.quantile(distances, 0.95)),
    )


@torch.no_grad()
def evaluate_cem_mpc(
    dynamics_model: nn.Module,
    reward_model: RewardModel,
    env_id: str,
    variants: list[str],
    seed: int,
    action_smoothing: float,
    history_length: int,
    config: PlanningEvalConfig,
    device: torch.device | str,
    use_oracle_context: bool = False,
    context_templates: tuple[MechanismTemplate, ...] = (),
    coverage_stats: PlannerCoverageStats | None = None,
) -> dict[str, float]:
    if not config.enabled:
        return {}
    if config.episodes <= 0:
        raise ValueError("planning episodes must be positive")
    if config.max_steps <= 0:
        raise ValueError("planning max_steps must be positive")
    if not variants:
        raise ValueError("planning variants must be non-empty")

    dynamics_model.eval()
    reward_model.eval()
    previous_planning_mode = getattr(dynamics_model, "planning_mode", None)
    if hasattr(dynamics_model, "set_planning_mode"):
        dynamics_model.set_planning_mode(True)
    rng = np.random.default_rng(seed)
    returns: list[float] = []
    lengths: list[int] = []
    ood_distances: list[float] = []
    ood_excesses: list[float] = []
    try:
        for episode in range(config.episodes):
            variant = variants[episode % len(variants)]
            context = sample_context(variant, rng)
            (
                episode_return,
                episode_length,
                episode_ood_distances,
                episode_ood_excesses,
            ) = _run_planned_episode(
                dynamics_model=dynamics_model,
                reward_model=reward_model,
                env_id=env_id,
                context=context,
                seed=seed + 10_007 * episode,
                action_smoothing=action_smoothing,
                history_length=history_length,
                config=config,
                device=device,
                use_oracle_context=use_oracle_context,
                context_templates=context_templates,
                coverage_stats=coverage_stats,
            )
            returns.append(episode_return)
            lengths.append(episode_length)
            ood_distances.extend(episode_ood_distances)
            ood_excesses.extend(episode_ood_excesses)
    finally:
        if hasattr(dynamics_model, "set_planning_mode") and previous_planning_mode is not None:
            dynamics_model.set_planning_mode(bool(previous_planning_mode))
    metrics = {
        "planner_return_mean": float(np.mean(returns)),
        "planner_return_std": float(np.std(returns)),
        "planner_length_mean": float(np.mean(lengths)),
    }
    if ood_distances:
        metrics["planner_ood_mean"] = float(np.mean(ood_distances))
        metrics["planner_ood_p95"] = float(np.quantile(ood_distances, 0.95))
        metrics["planner_ood_excess_mean"] = float(np.mean(ood_excesses))
        metrics["planner_coverage_train_p95"] = float(coverage_stats.train_p95) if coverage_stats else 0.0
    return metrics


def _run_planned_episode(
    dynamics_model: nn.Module,
    reward_model: RewardModel,
    env_id: str,
    context: np.ndarray,
    seed: int,
    action_smoothing: float,
    history_length: int,
    config: PlanningEvalConfig,
    device: torch.device | str,
    use_oracle_context: bool = False,
    context_templates: tuple[MechanismTemplate, ...] = (),
    coverage_stats: PlannerCoverageStats | None = None,
) -> tuple[float, int, list[float], list[float]]:
    env = gym.make(env_id)
    try:
        apply_parameter_context(env, context)
        obs, _ = env.reset(seed=seed)
        obs = np.asarray(obs, dtype=np.float32)
        action_low = np.asarray(env.action_space.low, dtype=np.float32)
        action_high = np.asarray(env.action_space.high, dtype=np.float32)
        previous_action = np.zeros_like(action_low, dtype=np.float32)
        history_states = np.repeat(obs[None, :], history_length, axis=0).astype(np.float32)
        history_actions = np.repeat(previous_action[None, :], history_length, axis=0).astype(np.float32)
        total_return = 0.0
        planner_config = CEMPlannerConfig(
            horizon=config.horizon,
            candidates=config.candidates,
            elites=config.elites,
            iterations=config.iterations,
            uncertainty_weight=config.uncertainty_weight,
            model_bonus_weight=config.model_bonus_weight,
            certified_risk_weight=config.certified_risk_weight,
        )
        rng = np.random.default_rng(seed + 1)
        model_context = (
            align_raw_context_to_templates(context, context_templates)
            if use_oracle_context
            else None
        )
        ood_distances: list[float] = []
        ood_excesses: list[float] = []
        for step in range(config.max_steps):
            action = plan_cem_action(
                model=dynamics_model,
                state=obs,
                action_low=action_low,
                action_high=action_high,
                reward_fn=lambda states, actions, next_states: reward_model(
                    states,
                    actions,
                    next_states,
                ),
                config=planner_config,
                device=device,
                history_states=history_states,
                history_actions=history_actions,
                model_context=model_context,
            )
            action = np.clip(action.astype(np.float32), action_low, action_high)
            if coverage_stats is not None:
                distance = coverage_stats.distance(obs, action)
                ood_distances.append(distance)
                ood_excesses.append(max(0.0, distance - coverage_stats.train_p95))
            effective_action = apply_action_context(action, previous_action, context, rng)
            env_next, reward, terminated, truncated, _ = env.step(effective_action)
            env_next = np.asarray(env_next, dtype=np.float32)
            next_obs = apply_transition_context(
                state=obs,
                next_state=env_next,
                context=context,
                rng=rng,
            )
            total_return += float(reward)
            previous_action = action
            obs = next_obs
            history_states = np.concatenate([history_states[1:], obs[None, :]], axis=0)
            history_actions = np.concatenate([history_actions[1:], action[None, :]], axis=0)
            if bool(terminated or truncated):
                return total_return, step + 1, ood_distances, ood_excesses
        return total_return, config.max_steps, ood_distances, ood_excesses
    finally:
        env.close()


def align_raw_context_to_templates(
    context: np.ndarray,
    templates: tuple[MechanismTemplate, ...],
) -> np.ndarray:
    if not templates:
        return context.astype(np.float32)
    name_to_index = {name: index for index, name in enumerate(CONTEXT_NAMES)}
    aligned = np.zeros(len(templates), dtype=np.float32)
    for column, template in enumerate(templates):
        if template.name in name_to_index:
            aligned[column] = float(context[name_to_index[template.name]])
        elif template.name == "actuation":
            aligned[column] = 1.0
    return aligned
