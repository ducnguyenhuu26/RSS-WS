from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from onelife.mujoco_dataset import MuJoCoTransitions


CONTEXT_NAMES: tuple[str, ...] = (
    "wind",
    "friction",
    "mass",
    "damping",
    "delay",
    "sticky",
    "impulse",
    "gravity",
)


@dataclass(frozen=True)
class MuJoCoExtensionConfig:
    env_id: str
    num_steps: int
    seed: int = 0
    variant: str = "all"
    num_contexts: int = 8
    max_episode_steps: int | None = None
    action_policy: str = "smooth_random"
    action_smoothing: float = 0.85


def collect_mujoco_extension_dataset(config: MuJoCoExtensionConfig) -> MuJoCoTransitions:
    """Collect MuJoCo transitions with DUC hidden-mechanism context labels.

    Context labels are generated from a compact prior support. Some mechanisms
    alter real env actions/parameters; lightweight disturbances are applied at
    the observation-transition level to keep the extension portable across
    Gymnasium MuJoCo tasks.
    """

    if config.num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if config.num_contexts <= 0:
        raise ValueError("num_contexts must be positive")
    rng = np.random.default_rng(config.seed)
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    next_states: list[np.ndarray] = []
    rewards: list[float] = []
    dones: list[bool] = []
    contexts: list[np.ndarray] = []
    steps_per_context = max(1, int(np.ceil(config.num_steps / config.num_contexts)))
    collected = 0
    for context_id in range(config.num_contexts):
        context = sample_context(config.variant, rng)
        env = gym.make(config.env_id)
        try:
            apply_parameter_context(env, context)
            obs, _ = env.reset(seed=config.seed + context_id)
            obs = np.asarray(obs, dtype=np.float32)
            previous_action = np.zeros(env.action_space.shape, dtype=np.float32)
            for _ in range(steps_per_context):
                if collected >= config.num_steps:
                    break
                raw_action = sample_action(
                    env,
                    rng,
                    previous_action=previous_action,
                    policy=config.action_policy,
                    smoothing=config.action_smoothing,
                )
                effective_action = apply_action_context(raw_action, previous_action, context, rng)
                env_next, reward, terminated, truncated, _ = env.step(effective_action)
                env_next = np.asarray(env_next, dtype=np.float32)
                disturbed_next = apply_transition_context(
                    state=obs,
                    next_state=env_next,
                    context=context,
                    rng=rng,
                )
                states.append(obs.astype(np.float32))
                actions.append(raw_action.astype(np.float32))
                next_states.append(disturbed_next.astype(np.float32))
                rewards.append(float(reward))
                done = bool(terminated or truncated)
                dones.append(done)
                contexts.append(context.astype(np.float32))
                collected += 1
                previous_action = raw_action
                obs = disturbed_next
                if done:
                    obs, _ = env.reset(seed=config.seed + context_id + collected)
                    obs = np.asarray(obs, dtype=np.float32)
                    previous_action = np.zeros(env.action_space.shape, dtype=np.float32)
        finally:
            env.close()
    return MuJoCoTransitions(
        states=np.asarray(states, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        next_states=np.asarray(next_states, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        dones=np.asarray(dones, dtype=np.bool_),
        contexts=np.asarray(contexts, dtype=np.float32),
        context_names=CONTEXT_NAMES,
    )


def sample_context(variant: str, rng: np.random.Generator) -> np.ndarray:
    enabled = enabled_context_names(variant)
    context = np.zeros(len(CONTEXT_NAMES), dtype=np.float32)
    ranges = {
        "wind": (-0.5, 0.5),
        "friction": (-0.4, 0.4),
        "mass": (-0.25, 0.25),
        "damping": (-0.35, 0.35),
        "delay": (0.0, 2.0),
        "sticky": (0.0, 0.35),
        "impulse": (-0.6, 0.6),
        "gravity": (-0.2, 0.2),
    }
    for index, name in enumerate(CONTEXT_NAMES):
        if name in enabled:
            low, high = ranges[name]
            context[index] = float(rng.uniform(low, high))
    return context


def enabled_context_names(variant: str) -> set[str]:
    raw = variant.strip()
    if raw == "all":
        return set(CONTEXT_NAMES)
    if raw in {"", "none"}:
        return set()
    enabled = {name.strip() for name in raw.split("+") if name.strip()}
    unknown = enabled.difference(CONTEXT_NAMES)
    if unknown:
        known = ", ".join(CONTEXT_NAMES)
        bad = ", ".join(sorted(unknown))
        raise ValueError(f"unknown MuJoCo extension context(s): {bad}; known contexts: {known}")
    return enabled


def context_value(context: np.ndarray, name: str) -> float:
    return float(context[CONTEXT_NAMES.index(name)])


def sample_action(
    env: gym.Env,
    rng: np.random.Generator,
    previous_action: np.ndarray,
    policy: str,
    smoothing: float,
) -> np.ndarray:
    low = np.asarray(env.action_space.low, dtype=np.float32)
    high = np.asarray(env.action_space.high, dtype=np.float32)
    random_action = rng.uniform(low=low, high=high).astype(np.float32)
    if policy == "random":
        return random_action
    if policy == "zero":
        return np.zeros_like(random_action)
    if policy != "smooth_random":
        raise ValueError(f"unknown action_policy={policy!r}")
    return np.clip(
        smoothing * previous_action + (1.0 - smoothing) * random_action,
        low,
        high,
    ).astype(np.float32)


def apply_action_context(
    action: np.ndarray,
    previous_action: np.ndarray,
    context: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    effective = action.copy()
    delay = context_value(context, "delay")
    if delay >= 0.5:
        mix = min(1.0, delay / 2.0)
        effective = (1.0 - mix) * effective + mix * previous_action
    sticky = context_value(context, "sticky")
    if sticky > 0 and rng.random() < sticky:
        effective = previous_action.copy()
    return effective.astype(np.float32)


def apply_parameter_context(env: gym.Env, context: np.ndarray) -> None:
    unwrapped = env.unwrapped
    model = getattr(unwrapped, "model", None)
    if model is None:
        return
    mass = context_value(context, "mass")
    if hasattr(model, "body_mass"):
        model.body_mass[:] = np.maximum(1e-6, model.body_mass * (1.0 + mass))
    damping = context_value(context, "damping")
    if hasattr(model, "dof_damping"):
        model.dof_damping[:] = np.maximum(0.0, model.dof_damping * (1.0 + damping))
    gravity = context_value(context, "gravity")
    if hasattr(model, "opt") and hasattr(model.opt, "gravity"):
        model.opt.gravity[2] = model.opt.gravity[2] * (1.0 + gravity)
    friction = context_value(context, "friction")
    if hasattr(model, "geom_friction"):
        model.geom_friction[:] = np.maximum(1e-6, model.geom_friction * (1.0 + friction))


def apply_transition_context(
    state: np.ndarray,
    next_state: np.ndarray,
    context: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    adjusted = np.asarray(next_state, dtype=np.float32).copy()
    dim = adjusted.shape[0]
    split = max(1, dim // 2)
    velocity = np.arange(split, dim)
    if velocity.size == 0:
        velocity = np.arange(dim)

    wind = context_value(context, "wind")
    if abs(wind) > 1e-6:
        affected = velocity[: min(2, len(velocity))]
        adjusted[affected] += 0.05 * wind

    friction = context_value(context, "friction")
    if abs(friction) > 1e-6:
        delta = adjusted[velocity] - state[velocity]
        adjusted[velocity] = state[velocity] + delta * max(0.2, 1.0 - 0.35 * abs(friction))

    sticky = context_value(context, "sticky")
    if sticky > 0 and rng.random() < sticky:
        adjusted = state + 0.15 * (adjusted - state)

    impulse = context_value(context, "impulse")
    if abs(impulse) > 1e-6 and rng.random() < 0.08 * min(1.0, abs(impulse) + 0.1):
        affected = velocity[: min(3, len(velocity))]
        adjusted[affected] += rng.normal(loc=impulse, scale=0.05, size=len(affected)).astype(np.float32)
    return adjusted.astype(np.float32)
