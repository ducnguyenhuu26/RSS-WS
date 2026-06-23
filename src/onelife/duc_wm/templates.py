from __future__ import annotations

from dataclasses import dataclass

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
    description: str = ""

    def validate(self, state_dim: int, action_dim: int) -> None:
        if not self.output_indices:
            raise ValueError(f"mechanism {self.name!r} must affect at least one state dim")
        if self.scale <= 0:
            raise ValueError(f"mechanism {self.name!r} scale must be positive")
        if self.prior_std <= 0:
            raise ValueError(f"mechanism {self.name!r} prior_std must be positive")
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
    A real LLM prior file can refine the same fields later; the model code only
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
            description="agent action induces generalized velocity/body changes",
        ),
        MechanismTemplate(
            name="wind",
            state_indices=pos + vel,
            action_indices=(),
            output_indices=xy_vel,
            scale=0.75,
            prior_std=0.5,
            description="external field produces persistent horizontal drift",
        ),
        MechanismTemplate(
            name="friction",
            state_indices=contact_like,
            action_indices=all_action,
            output_indices=vel,
            scale=0.8,
            prior_std=0.6,
            description="contact/friction changes damp or amplify velocity response",
        ),
        MechanismTemplate(
            name="mass",
            state_indices=all_state,
            action_indices=all_action,
            output_indices=vel,
            scale=0.55,
            prior_std=0.4,
            description="mass/inertia shift changes acceleration induced by control",
        ),
        MechanismTemplate(
            name="damping",
            state_indices=vel,
            action_indices=all_action,
            output_indices=vel,
            scale=0.55,
            prior_std=0.4,
            description="damping dissipates velocity and changes rollout stability",
        ),
        MechanismTemplate(
            name="delay",
            state_indices=all_state,
            action_indices=all_action,
            output_indices=vel,
            scale=0.6,
            prior_std=0.5,
            description="actuator delay makes current state respond to stale control",
        ),
        MechanismTemplate(
            name="sticky",
            state_indices=all_state,
            action_indices=all_action,
            output_indices=all_state,
            scale=0.5,
            prior_std=0.4,
            description="transition becomes partially stuck near the previous state",
        ),
        MechanismTemplate(
            name="impulse",
            state_indices=pos + vel,
            action_indices=(),
            output_indices=vel,
            scale=0.7,
            prior_std=0.4,
            description="rare unmodeled force causes sudden velocity change",
        ),
        MechanismTemplate(
            name="gravity",
            state_indices=pos + vel,
            action_indices=(),
            output_indices=vel,
            scale=0.45,
            prior_std=0.3,
            description="gravity shift changes passive acceleration and balance",
        ),
    )


def prior_tensors(
    templates: tuple[MechanismTemplate, ...],
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    means = torch.tensor([item.prior_mean for item in templates], dtype=torch.float32, device=device)
    stds = torch.tensor([item.prior_std for item in templates], dtype=torch.float32, device=device)
    scales = torch.tensor([item.scale for item in templates], dtype=torch.float32, device=device)
    return means, stds, scales
