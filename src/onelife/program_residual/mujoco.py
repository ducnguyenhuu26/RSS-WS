from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from onelife.mujoco_dataset import MuJoCoTransitions

from .core import TransitionBatch


PolicyFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class MuJoCoCollectionConfig:
    env_id: str
    num_steps: int
    seed: int = 0
    render_mode: str | None = None


def _as_float_array(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1:
        return array.reshape(-1)
    return array


def collect_transitions_from_env(
    env: gym.Env,
    num_steps: int,
    policy: PolicyFn | None = None,
    seed: int = 0,
) -> TransitionBatch:
    """Collect continuous-control transitions as a training batch."""
    return transitions_to_batch(
        collect_dataset_from_env(
            env=env,
            num_steps=num_steps,
            policy=policy,
            seed=seed,
        )
    )


def collect_dataset_from_env(
    env: gym.Env,
    num_steps: int,
    policy: PolicyFn | None = None,
    seed: int = 0,
) -> MuJoCoTransitions:
    """Collect continuous-control transitions from a Gymnasium-style env."""
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")

    reset_result = env.reset(seed=seed)
    observation = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    next_states: list[np.ndarray] = []

    for _ in range(num_steps):
        state = _as_float_array(observation)
        if policy is None:
            action = _as_float_array(env.action_space.sample())
        else:
            action = _as_float_array(policy(state))

        step_result = env.step(action)
        if len(step_result) == 5:
            next_observation, _reward, terminated, truncated, _info = step_result
            done = bool(terminated or truncated)
        elif len(step_result) == 4:
            next_observation, _reward, done, _info = step_result
            done = bool(done)
        else:
            raise ValueError("env.step must return 4 or 5 values")

        states.append(state)
        actions.append(action)
        next_states.append(_as_float_array(next_observation))

        if done:
            reset_result = env.reset()
            observation = (
                reset_result[0] if isinstance(reset_result, tuple) else reset_result
            )
        else:
            observation = next_observation

    return MuJoCoTransitions(
        states=np.stack(states).astype(np.float32),
        actions=np.stack(actions).astype(np.float32),
        next_states=np.stack(next_states).astype(np.float32),
    )


def collect_mujoco_transitions(
    config: MuJoCoCollectionConfig,
    policy: PolicyFn | None = None,
) -> TransitionBatch:
    """Create a Gymnasium MuJoCo env by id and collect transitions."""
    return transitions_to_batch(collect_mujoco_dataset(config=config, policy=policy))


def collect_mujoco_dataset(
    config: MuJoCoCollectionConfig,
    policy: PolicyFn | None = None,
) -> MuJoCoTransitions:
    """Create a Gymnasium MuJoCo env by id and collect a shared dataset."""
    env = gym.make(config.env_id, render_mode=config.render_mode)
    try:
        return collect_dataset_from_env(
            env=env,
            num_steps=config.num_steps,
            policy=policy,
            seed=config.seed,
        )
    finally:
        env.close()


def transitions_to_batch(transitions: MuJoCoTransitions) -> TransitionBatch:
    states, actions, next_states = transitions.to_torch()
    return TransitionBatch(states=states, actions=actions, next_states=next_states)
