from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class MechanismTemplate:
    """LLM/prior-provided structural hint for one causal mechanism."""

    name: str
    state_indices: tuple[int, ...]
    action_indices: tuple[int, ...]
    output_indices: tuple[int, ...]
    scale: float = 1.0
    prior_mean: float = 0.0
    prior_std: float = 1.0
    prior_confidence: float = 0.5
    timescale: str = "slow"
    reward_relevance: str = ""
    description: str = ""

    def validate(self, state_dim: int, action_dim: int) -> None:
        if not self.output_indices:
            raise ValueError(f"mechanism {self.name!r} must affect at least one state dim")
        if self.scale <= 0:
            raise ValueError(f"mechanism {self.name!r} scale must be positive")
        if self.prior_std <= 0:
            raise ValueError(f"mechanism {self.name!r} prior_std must be positive")
        if not 0.0 <= self.prior_confidence <= 1.0:
            raise ValueError(f"mechanism {self.name!r} prior_confidence must be in [0, 1]")
        if self.timescale not in {"slow", "event", "unknown"}:
            raise ValueError(
                f"mechanism {self.name!r} timescale must be slow, event, or unknown"
            )
        for index in self.state_indices + self.output_indices:
            if index < 0 or index >= state_dim:
                raise ValueError(
                    f"mechanism {self.name!r} state/output index {index} "
                    f"is outside state_dim={state_dim}"
                )
        for index in self.action_indices:
            if index < 0 or index >= action_dim:
                raise ValueError(
                    f"mechanism {self.name!r} action index {index} "
                    f"is outside action_dim={action_dim}"
                )


def default_mujoco_templates(
    env_id: str,
    state_dim: int,
    action_dim: int,
) -> tuple[MechanismTemplate, ...]:
    """Return a compact DUC-WM mechanism prior for MuJoCo-style vectors.

    These templates are the deterministic fallback for the offline LLM prior.
    A real LLM prior file can refine the same fields; the model code only
    depends on masks, scales, and prior ranges.
    """

    all_state = tuple(range(state_dim))
    all_action = tuple(range(action_dim))
    split = max(1, state_dim // 2)
    pos = tuple(range(split))
    vel = tuple(range(split, state_dim)) or all_state
    xy_vel = vel[: min(2, len(vel))] or all_state[: min(2, state_dim)]
    contact_like = vel if "ant" in env_id.lower() or "hopper" in env_id.lower() else all_state

    return (
        MechanismTemplate(
            name="actuation",
            state_indices=all_state,
            action_indices=all_action,
            output_indices=vel,
            scale=1.25,
            prior_std=0.7,
            prior_confidence=0.65,
            timescale="slow",
            reward_relevance="baseline control authority and velocity response",
            description="agent action induces generalized velocity/body changes",
        ),
        MechanismTemplate(
            name="wind",
            state_indices=pos + vel,
            action_indices=(),
            output_indices=xy_vel,
            scale=0.75,
            prior_std=0.5,
            prior_confidence=0.55,
            timescale="slow",
            reward_relevance="persistent drift changes forward and lateral velocity",
            description="external field produces persistent horizontal drift",
        ),
        MechanismTemplate(
            name="friction",
            state_indices=contact_like,
            action_indices=all_action,
            output_indices=vel,
            scale=0.8,
            prior_std=0.6,
            prior_confidence=0.6,
            timescale="slow",
            reward_relevance="contact slip changes gait and action-to-motion transfer",
            description="contact/friction changes damp or amplify velocity response",
        ),
        MechanismTemplate(
            name="mass",
            state_indices=all_state,
            action_indices=all_action,
            output_indices=vel,
            scale=0.55,
            prior_std=0.4,
            prior_confidence=0.55,
            timescale="slow",
            reward_relevance="inertia changes acceleration induced by the same action",
            description="mass/inertia shift changes acceleration induced by control",
        ),
        MechanismTemplate(
            name="damping",
            state_indices=vel,
            action_indices=all_action,
            output_indices=vel,
            scale=0.55,
            prior_std=0.4,
            prior_confidence=0.65,
            timescale="slow",
            reward_relevance="passive velocity loss affects rollout stability",
            description="damping dissipates velocity and changes rollout stability",
        ),
        MechanismTemplate(
            name="delay",
            state_indices=all_state,
            action_indices=all_action,
            output_indices=vel,
            scale=0.6,
            prior_std=0.5,
            prior_confidence=0.5,
            timescale="event",
            reward_relevance="stale actuation shifts short-horizon control response",
            description="actuator delay makes current state respond to stale control",
        ),
        MechanismTemplate(
            name="sticky",
            state_indices=all_state,
            action_indices=all_action,
            output_indices=all_state,
            scale=0.5,
            prior_std=0.4,
            prior_confidence=0.45,
            timescale="event",
            reward_relevance="partial no-op transitions cause abrupt rollout mismatch",
            description="transition becomes partially stuck near the previous state",
        ),
        MechanismTemplate(
            name="impulse",
            state_indices=pos + vel,
            action_indices=(),
            output_indices=vel,
            scale=0.7,
            prior_std=0.4,
            prior_confidence=0.35,
            timescale="event",
            reward_relevance="rare external kicks perturb velocity and balance",
            description="rare unmodeled force causes sudden velocity change",
        ),
        MechanismTemplate(
            name="gravity",
            state_indices=pos + vel,
            action_indices=(),
            output_indices=vel,
            scale=0.45,
            prior_std=0.3,
            prior_confidence=0.65,
            timescale="slow",
            reward_relevance="passive acceleration changes balance and fall risk",
            description="gravity shift changes passive acceleration and balance",
        ),
        MechanismTemplate(
            name="unknown",
            state_indices=all_state,
            action_indices=all_action,
            output_indices=all_state,
            scale=0.35,
            prior_std=0.25,
            prior_confidence=0.0,
            timescale="unknown",
            reward_relevance="absorbs transition shift not explained by named mechanisms",
            description="fallback residual slot for missing or wrong prior mechanisms",
        ),
    )


def prior_tensors(
    templates: tuple[MechanismTemplate, ...],
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    means = torch.tensor([item.prior_mean for item in templates], dtype=torch.float32, device=device)
    stds = torch.tensor([item.prior_std for item in templates], dtype=torch.float32, device=device)
    scales = torch.tensor([item.scale for item in templates], dtype=torch.float32, device=device)
    confidences = torch.tensor(
        [item.prior_confidence for item in templates],
        dtype=torch.float32,
        device=device,
    )
    return means, stds, scales, confidences


def randomize_mechanism_templates(
    templates: tuple[MechanismTemplate, ...],
    state_dim: int,
    action_dim: int,
    seed: int,
) -> tuple[MechanismTemplate, ...]:
    """Return a random-mask prior with matched mechanism count and scales.

    Names, timescale, and strength priors are preserved so the comparison
    isolates the value of LLM/default semantic masks rather than model size.
    The unknown slot remains dense because it is a shared fallback mechanism.
    """

    rng = np.random.default_rng(seed)
    randomized: list[MechanismTemplate] = []
    all_state = np.arange(state_dim)
    all_action = np.arange(action_dim)
    for template in templates:
        if template.name == "unknown" or template.timescale == "unknown":
            randomized.append(template)
            continue
        state_count = max(1, min(state_dim, len(template.state_indices)))
        action_count = min(action_dim, len(template.action_indices))
        output_count = max(1, min(state_dim, len(template.output_indices)))
        state_indices = tuple(
            sorted(int(index) for index in rng.choice(all_state, size=state_count, replace=False))
        )
        action_indices = tuple(
            sorted(int(index) for index in rng.choice(all_action, size=action_count, replace=False))
        )
        output_indices = tuple(
            sorted(int(index) for index in rng.choice(all_state, size=output_count, replace=False))
        )
        randomized.append(
            MechanismTemplate(
                name=template.name,
                state_indices=state_indices,
                action_indices=action_indices,
                output_indices=output_indices,
                scale=template.scale,
                prior_mean=template.prior_mean,
                prior_std=template.prior_std,
                prior_confidence=template.prior_confidence,
                timescale=template.timescale,
                reward_relevance="random-mask control prior for ablation",
                description=f"randomized mask ablation for {template.name}",
            )
        )
    return tuple(randomized)


def generic_mechanism_templates(
    state_dim: int,
    action_dim: int,
    count: int = 8,
    include_unknown: bool = True,
) -> tuple[MechanismTemplate, ...]:
    """Return non-semantic dense templates for the no-LLM ablation."""

    all_state = tuple(range(state_dim))
    all_action = tuple(range(action_dim))
    split = max(1, state_dim // 2)
    vel = tuple(range(split, state_dim)) or all_state
    templates: list[MechanismTemplate] = []
    for index in range(max(1, count)):
        output = vel if index % 2 == 0 else all_state
        templates.append(
            MechanismTemplate(
                name=f"generic_{index}",
                state_indices=all_state,
                action_indices=all_action,
                output_indices=output,
                scale=0.75,
                prior_std=0.75,
                prior_confidence=0.0,
                timescale="slow" if index < count // 2 else "event",
                reward_relevance="non-semantic mechanism ablation",
                description="dense non-semantic mechanism without LLM structure",
            )
        )
    if include_unknown:
        templates.append(
            MechanismTemplate(
                name="unknown",
                state_indices=all_state,
                action_indices=all_action,
                output_indices=all_state,
                scale=0.35,
                prior_std=0.25,
                prior_confidence=0.0,
                timescale="unknown",
                reward_relevance="absorbs missing non-semantic structure",
                description="fallback slot for no-LLM ablation",
            )
        )
    return tuple(templates)


def remove_unknown_template(
    templates: tuple[MechanismTemplate, ...],
) -> tuple[MechanismTemplate, ...]:
    filtered = tuple(
        template
        for template in templates
        if template.name != "unknown" and template.timescale != "unknown"
    )
    if not filtered:
        raise ValueError("cannot remove unknown from an empty template bank")
    return filtered


def wrong_mechanism_templates(
    templates: tuple[MechanismTemplate, ...],
) -> tuple[MechanismTemplate, ...]:
    """Return a deterministic wrong-prior ablation by rotating semantic masks."""

    semantic = [
        template
        for template in templates
        if template.name != "unknown" and template.timescale != "unknown"
    ]
    unknown = [
        template
        for template in templates
        if template.name == "unknown" or template.timescale == "unknown"
    ]
    if len(semantic) <= 1:
        return templates
    rotated = semantic[1:] + semantic[:1]
    wrong: list[MechanismTemplate] = []
    for original, donor in zip(semantic, rotated, strict=True):
        wrong.append(
            MechanismTemplate(
                name=original.name,
                state_indices=donor.state_indices,
                action_indices=donor.action_indices,
                output_indices=donor.output_indices,
                scale=original.scale,
                prior_mean=original.prior_mean,
                prior_std=original.prior_std,
                prior_confidence=original.prior_confidence,
                timescale=original.timescale,
                reward_relevance="wrong-prior ablation with rotated mechanism masks",
                description=f"{original.name} assigned masks from {donor.name}",
            )
        )
    return tuple(wrong + unknown)
