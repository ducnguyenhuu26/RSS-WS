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
from .planner import CEMPlannerConfig, plan_cem_action
from .reward_model import RewardModel


@dataclass(frozen=True)
class PlanningEvalConfig:
    enabled: bool = False
    episodes: int = 3
    max_steps: int = 200
    horizon: int = 15
    candidates: int = 256
    elites: int = 32
    iterations: int = 3
    uncertainty_weight: float = 0.05


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
    rng = np.random.default_rng(seed)
    returns: list[float] = []
    lengths: list[int] = []
    for episode in range(config.episodes):
        variant = variants[episode % len(variants)]
        context = sample_context(variant, rng)
        episode_return, episode_length = _run_planned_episode(
            dynamics_model=dynamics_model,
            reward_model=reward_model,
            env_id=env_id,
            context=context,
            seed=seed + 10_007 * episode,
            action_smoothing=action_smoothing,
            history_length=history_length,
            config=config,
            device=device,
        )
        returns.append(episode_return)
        lengths.append(episode_length)
    return {
        "planner_return_mean": float(np.mean(returns)),
        "planner_return_std": float(np.std(returns)),
        "planner_length_mean": float(np.mean(lengths)),
    }


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
) -> tuple[float, int]:
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
        )
        rng = np.random.default_rng(seed + 1)
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
            )
            action = np.clip(action.astype(np.float32), action_low, action_high)
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
                return total_return, step + 1
        return total_return, config.max_steps
    finally:
        env.close()
