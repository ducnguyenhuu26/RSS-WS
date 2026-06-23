from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .templates import MechanismTemplate, default_mujoco_templates


@dataclass(frozen=True)
class DUCPriorPrompt:
    env_id: str
    state_dim: int
    action_dim: int
    prompt: str
    fallback_templates: tuple[MechanismTemplate, ...]


def build_duc_mujoco_prior_prompt(
    env_id: str,
    state_dim: int,
    action_dim: int,
) -> DUCPriorPrompt:
    templates = default_mujoco_templates(env_id, state_dim, action_dim)
    profile = mujoco_environment_profile(env_id, state_dim, action_dim)
    mechanism_lines = "\n".join(
        f"- {template.name}: {template.description}; "
        f"state_indices={list(template.state_indices)}, "
        f"action_indices={list(template.action_indices)}, "
        f"output_indices={list(template.output_indices)}, "
        f"scale={template.scale}, prior_std={template.prior_std}"
        for template in templates
    )
    prompt = f"""
You are constructing the offline mechanism prior for DUC-WM.

DUC-WM is a hidden-mechanism world model for continuous-control MuJoCo tasks.
It represents transition deltas as:

Delta x_t = sum_j alpha_j M_j(x_t, a_t) + epsilon_t.

Your job is not to write executable simulator code. Your job is to refine the
mechanism templates used by small neural modules. Return only strict JSON.

Environment:
{profile}

Available flat vector interface:
- state_dim = {state_dim}
- action_dim = {action_dim}
- state coordinates are accessible only by integer indices 0..{state_dim - 1}
- action coordinates are accessible only by integer indices 0..{action_dim - 1}

Initial safe mechanism templates:
{mechanism_lines}

Return a JSON object with this schema:
{{
  "env_id": "{env_id}",
  "templates": [
    {{
      "name": "wind",
      "state_indices": [0, 1],
      "action_indices": [],
      "output_indices": [2, 3],
      "scale": 0.75,
      "prior_mean": 0.0,
      "prior_std": 0.5,
      "description": "short mechanism explanation"
    }}
  ]
}}

Rules:
- Use only valid integer indices within the given state/action dimensions.
- Prefer 6 to 10 mechanisms.
- Include actuation plus only mechanisms plausible for this specific env.
- Keep output_indices sparse when possible.
- Do not include arbitrary symbolic laws.
- Do not claim certainty; use scale/prior_std to encode uncertainty.
- Return JSON only, no markdown.
""".strip()
    return DUCPriorPrompt(
        env_id=env_id,
        state_dim=state_dim,
        action_dim=action_dim,
        prompt=prompt,
        fallback_templates=templates,
    )


def mujoco_environment_profile(env_id: str, state_dim: int, action_dim: int) -> str:
    key = env_id.lower()
    common = f"{env_id}: flat MuJoCo observation with {state_dim} state dims and {action_dim} action dims."
    if "swimmer" in key:
        return (
            common
            + " A low-dimensional chain swimmer. Forward motion depends on body bending, "
            "joint velocities, fluid-like drag, and smooth actuation."
        )
    if "reacher" in key:
        return (
            common
            + " A planar reaching arm. Reward is target-distance dominated, so mechanisms "
            "affect end-effector position, angular motion, and action precision."
        )
    if "pusher" in key:
        return (
            common
            + " A manipulation task with arm-object contact. Contact, friction, object "
            "sliding, and delayed actuation are especially plausible mechanisms."
        )
    if "hopper" in key:
        return (
            common
            + " A single-legged locomotion task. Balance, contact friction, damping, "
            "gravity, and sticky contact can strongly affect survival and forward motion."
        )
    if "walker2d" in key:
        return (
            common
            + " A two-legged locomotion task. Gait, contact, friction, mass, damping, "
            "gravity, and action delay affect forward velocity and fall risk."
        )
    if "halfcheetah" in key:
        return (
            common
            + " A planar fast locomotion task. Forward velocity, torso/joint velocity, "
            "actuation strength, damping, mass, and action delay are central."
        )
    if "ant" in key:
        return (
            common
            + " A high-dimensional quadruped. Multiple legs create contact-rich dynamics; "
            "wind, friction, impulse forces, delay, mass, and damping are plausible."
        )
    if "inverteddoublependulum" in key:
        return (
            common
            + " A balance task with coupled pendulum angles. Gravity, damping, and "
            "actuation delay affect stability and recovery."
        )
    if "invertedpendulum" in key:
        return (
            common
            + " A simple balance task. Gravity, damping, and action delay dominate the "
            "hidden-mechanism space."
        )
    return common + " Use conservative generic mechanisms: actuation, damping, delay, friction, and mild external force."


def templates_from_llm_json(
    text: str,
    state_dim: int,
    action_dim: int,
) -> tuple[MechanismTemplate, ...]:
    raw = json.loads(text)
    templates_raw = raw.get("templates")
    if not isinstance(templates_raw, list):
        raise ValueError("LLM prior JSON must contain a list field named 'templates'")
    templates: list[MechanismTemplate] = []
    for item in templates_raw:
        if not isinstance(item, dict):
            raise ValueError("each template must be a JSON object")
        template = MechanismTemplate(
            name=str(item["name"]),
            state_indices=tuple(int(index) for index in item.get("state_indices", ())),
            action_indices=tuple(int(index) for index in item.get("action_indices", ())),
            output_indices=tuple(int(index) for index in item.get("output_indices", ())),
            scale=float(item.get("scale", 1.0)),
            prior_mean=float(item.get("prior_mean", 0.0)),
            prior_std=float(item.get("prior_std", 1.0)),
            description=str(item.get("description", "")),
        )
        template.validate(state_dim, action_dim)
        templates.append(template)
    if not templates:
        raise ValueError("LLM prior JSON produced no templates")
    return tuple(templates)


def load_templates_from_json_file(
    path: str | Path,
    state_dim: int,
    action_dim: int,
) -> tuple[MechanismTemplate, ...]:
    return templates_from_llm_json(
        Path(path).read_text(encoding="utf-8"),
        state_dim=state_dim,
        action_dim=action_dim,
    )


def prompt_payload(prompt: DUCPriorPrompt) -> dict[str, Any]:
    return {
        "env_id": prompt.env_id,
        "state_dim": prompt.state_dim,
        "action_dim": prompt.action_dim,
        "prompt": prompt.prompt,
        "fallback_templates": [
            {
                "name": item.name,
                "state_indices": list(item.state_indices),
                "action_indices": list(item.action_indices),
                "output_indices": list(item.output_indices),
                "scale": item.scale,
                "prior_mean": item.prior_mean,
                "prior_std": item.prior_std,
                "description": item.description,
            }
            for item in prompt.fallback_templates
        ],
    }
