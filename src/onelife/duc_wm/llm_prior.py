from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from onelife.litellm_utils import LiteLlmMessage, LiteLlmParamsBase, LiteLlmRequest

from .templates import ALLOWED_LAW_TYPES, MechanismTemplate, default_mujoco_templates


@dataclass(frozen=True)
class DUCPriorPrompt:
    env_id: str
    state_dim: int
    action_dim: int
    prompt: str
    fallback_templates: tuple[MechanismTemplate, ...]


@dataclass(frozen=True)
class DUCLLMPriorConfig:
    provider: str = "openai"
    model_slug: str = "gpt-4.1-mini"
    api_key_env: str = "OPENAI_API_KEY"
    max_tokens: int = 3000


def build_duc_mujoco_prior_prompt(
    env_id: str,
    state_dim: int,
    action_dim: int,
) -> DUCPriorPrompt:
    templates = default_mujoco_templates(env_id, state_dim, action_dim)
    profile = mujoco_environment_profile(env_id, state_dim, action_dim)
    dimension_hint = mujoco_dimension_role_hint(env_id, state_dim, action_dim)
    mechanism_lines = "\n".join(
        f"- {template.name}: {template.description}; "
        f"state_indices={list(template.state_indices)}, "
        f"action_indices={list(template.action_indices)}, "
        f"output_indices={list(template.output_indices)}, "
        f"law_type={template.law_type}, law_gain={template.law_gain}, "
        f"scale={template.scale}, prior_std={template.prior_std}, "
        f"confidence={template.prior_confidence}, timescale={template.timescale}"
        for template in templates
    )
    prompt = f"""
You are the offline scientific prior builder for SimFutures-LP.

SimFutures-LP is a continuous-control world model and planner. It does not ask
an LLM to predict dynamics or actions directly. The LLM defines a safe portfolio
of executable law templates. Code compiles each template into a bounded tensor
law channel m_j(state, action, next_state). A neural model then learns a latent
posterior over which laws are valid and useful for reward-seeking planning.

Your answer must define which law templates should exist, what inputs they read,
what state coordinates they affect, which DSL law_type each law uses, whether it
is persistent or event-like, and how uncertain/confident the prior should be.

Do not write Python code. Do not write arbitrary symbolic equations. Do not
invent a full simulator. Return JSON only.

Environment profile:
{profile}

Flat vector interface:
- state_dim = {state_dim}
- action_dim = {action_dim}
- valid state indices: 0..{state_dim - 1}
- valid action indices: 0..{action_dim - 1}
- index role hint: {dimension_hint}

SimFutures-LP law interpretation:
- actuation: action-driven change in velocity/body coordinates.
- wind: persistent external drift or force.
- friction: contact or surface-dependent velocity damping.
- mass: changed inertia, so the same action causes different acceleration.
- damping: passive dissipation of velocity or angular velocity.
- delay: stale action influence or slow actuator response.
- sticky: partial no-op transition, stalling, or contact sticking.
- impulse: sparse unobserved force or contact kick.
- gravity: changed passive acceleration and balance stability.

Allowed law_type DSL values:
learned_residual, actuation, external_drift, velocity_damping, inertia_shift, action_delay, sticky_velocity, impulse, gravity_shift

Recommended mapping:
- actuation -> law_type="actuation"
- wind -> law_type="external_drift"
- friction or damping -> law_type="velocity_damping"
- mass -> law_type="inertia_shift"
- delay -> law_type="action_delay"
- sticky -> law_type="sticky_velocity"
- impulse -> law_type="impulse"
- gravity -> law_type="gravity_shift"
- unknown -> law_type="learned_residual"

Initial safe fallback templates:
{mechanism_lines}

Your task:
1. Keep law templates that are plausible for this exact environment.
2. Remove laws that are weak or redundant for this environment.
3. Choose law_type and law_gain for the executable law channel.
4. Adjust state_indices, action_indices, output_indices, scale, and prior_std.
5. Make output_indices sparse unless the law truly affects the whole body.
6. Encode uncertainty through prior_std and confidence, not through overclaiming.
7. Prefer laws that can separate high-reward rollouts from merely accurate rollouts.
8. Ensure actuation is present.
9. Use 6 to 10 mechanisms.

Return a JSON object with this exact schema:
{{
  "env_id": "{env_id}",
  "templates": [
    {{
      "name": "wind",
      "state_indices": [0, 1],
      "action_indices": [],
      "output_indices": [2, 3],
      "law_type": "external_drift",
      "law_gain": 0.05,
      "scale": 0.75,
      "prior_mean": 0.0,
      "prior_std": 0.5,
      "confidence": 0.65,
      "timescale": "slow",
      "reward_relevance": "how this law affects planning reward",
      "description": "short law explanation grounded in this environment"
    }}
  ]
}}

Hard rules:
- Use only valid integer indices within the given state/action dimensions.
- Use only mechanism names from the SimFutures-LP law interpretation list unless a
  new name is absolutely necessary.
- Do not include duplicate names.
- Do not include empty output_indices.
- Do not include arbitrary symbolic laws or executable code.
- law_type must be one of the allowed DSL values above.
- Use law_gain in [-0.5, 0.5] for most MuJoCo transition priors; prefer small magnitudes.
- Use timescale="slow" for persistent physics, "event" for transient mechanisms,
  and reserve "unknown" for a fallback residual slot if included.
- Use scale in [0.05, 2.0].
- Use prior_std in [0.05, 1.5].
- Use confidence in [0.0, 1.0].
- Return JSON only, no markdown, no prose outside the JSON.
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
            + " Task: produce forward swimming motion with smooth joint actuation. "
            "Reward-critical errors usually come from body bending, joint velocity, "
            "fluid-like drag, weak actuation, and persistent drift."
        )
    if "reacher" in key:
        return (
            common
            + " Task: move a planar reaching arm end-effector close to a target. Reward-critical "
            "errors are target-distance errors caused by angular dynamics, damping, "
            "delay, action precision, and mild endpoint disturbance."
        )
    if "pusher" in key:
        return (
            common
            + " Task: move an object through arm-object contact. Reward-critical errors "
            "come from contact mode changes, object sliding, friction, impulse, delay, "
            "and actuation mismatch."
        )
    if "hopper" in key:
        return (
            common
            + " Task: survive and move forward with a single leg. Reward-critical errors "
            "come from balance, ground contact, friction, gravity, damping, sticky "
            "contact, and action-to-velocity mismatch."
        )
    if "walker2d" in key:
        return (
            common
            + " Task: sustain a two-legged gait and move forward. Reward-critical errors "
            "come from gait phase, foot contact, friction, mass, damping, gravity, "
            "action delay, and fall-risk state dimensions."
        )
    if "halfcheetah" in key:
        return (
            common
            + " Task: maximize forward velocity without falling constraints dominating. "
            "Reward-critical errors come from torso/joint velocity, actuation strength, "
            "mass, damping, friction, and action delay."
        )
    if "ant" in key:
        return (
            common
            + " Task: coordinate a high-dimensional quadruped for forward motion. "
            "Reward-critical errors come from multi-leg contact, body velocity, wind, "
            "friction, impulse forces, delay, mass, damping, and fall-risk dimensions."
        )
    if "inverteddoublependulum" in key:
        return (
            common
            + " Task: keep a coupled pendulum balanced. Reward-critical errors come "
            "from small angular deviations, angular velocity, gravity, damping, "
            "and action delay."
        )
    if "invertedpendulum" in key:
        return (
            common
            + " Task: keep the pendulum upright. Reward-critical errors come from "
            "angle, angular velocity, gravity, damping, weak actuation, and delay."
        )
    return common + " Use conservative generic mechanisms: actuation, damping, delay, friction, and mild external force."


def mujoco_dimension_role_hint(env_id: str, state_dim: int, action_dim: int) -> str:
    split = max(1, state_dim // 2)
    pos = f"state[0:{split}] are usually position/angle/body-pose like"
    vel = f"state[{split}:{state_dim}] are usually velocity/angular-velocity like"
    action = f"action[0:{action_dim}] are control inputs"
    key = env_id.lower()
    if "reacher" in key:
        return (
            f"{pos}; {vel}; final observation entries may include target or "
            f"end-effector offset information; {action}."
        )
    if "pusher" in key:
        return (
            f"{pos}; {vel}; some coordinates may describe object position/contact; "
            f"{action}."
        )
    if "ant" in key:
        return (
            f"{pos}; {vel}; high-index entries often include leg/contact velocity "
            f"information; {action}."
        )
    return f"{pos}; {vel}; {action}."


def templates_from_llm_json(
    text: str,
    state_dim: int,
    action_dim: int,
) -> tuple[MechanismTemplate, ...]:
    raw = json.loads(_extract_json_object(text))
    templates_raw = raw.get("templates")
    if not isinstance(templates_raw, list):
        raise ValueError("LLM prior JSON must contain a list field named 'templates'")
    templates: list[MechanismTemplate] = []
    seen_names: set[str] = set()
    for item in templates_raw:
        if not isinstance(item, dict):
            raise ValueError("each template must be a JSON object")
        name = str(item["name"])
        if name in seen_names:
            raise ValueError(f"duplicate mechanism name {name!r}")
        seen_names.add(name)
        template = MechanismTemplate(
            name=name,
            state_indices=tuple(int(index) for index in item.get("state_indices", ())),
            action_indices=tuple(int(index) for index in item.get("action_indices", ())),
            output_indices=tuple(int(index) for index in item.get("output_indices", ())),
            law_type=str(item.get("law_type", _default_law_type_for_name(name))),
            law_gain=float(item.get("law_gain", item.get("coefficient_prior", 1.0))),
            scale=float(item.get("scale", 1.0)),
            prior_mean=float(item.get("prior_mean", 0.0)),
            prior_std=float(item.get("prior_std", 1.0)),
            prior_confidence=float(item.get("confidence", item.get("prior_confidence", 0.5))),
            timescale=str(item.get("timescale", _default_timescale_for_name(name))),
            reward_relevance=str(item.get("reward_relevance", "")),
            description=str(item.get("description", "")),
        )
        template.validate(state_dim, action_dim)
        if not 0.05 <= template.scale <= 2.0:
            raise ValueError(f"mechanism {template.name!r} scale outside [0.05, 2.0]")
        if not 0.05 <= template.prior_std <= 1.5:
            raise ValueError(f"mechanism {template.name!r} prior_std outside [0.05, 1.5]")
        templates.append(template)
    if not templates:
        raise ValueError("LLM prior JSON produced no templates")
    return tuple(templates)


def _default_law_type_for_name(name: str) -> str:
    mapping = {
        "actuation": "actuation",
        "wind": "external_drift",
        "friction": "velocity_damping",
        "mass": "inertia_shift",
        "damping": "velocity_damping",
        "delay": "action_delay",
        "sticky": "sticky_velocity",
        "impulse": "impulse",
        "gravity": "gravity_shift",
        "unknown": "learned_residual",
    }
    return mapping.get(name, "learned_residual")


def _default_timescale_for_name(name: str) -> str:
    if name == "unknown":
        return "unknown"
    if name in {"delay", "sticky", "impulse"}:
        return "event"
    return "slow"


def _extract_json_object(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("LLM prior response does not contain a JSON object")
    return text[start : end + 1]


def synthesize_templates_with_llm(
    prior_prompt: DUCPriorPrompt,
    state_dim: int,
    action_dim: int,
    config: DUCLLMPriorConfig,
) -> tuple[tuple[MechanismTemplate, ...], str]:
    api_key = os.environ.get(config.api_key_env)
    if not api_key:
        raise RuntimeError(f"missing API key environment variable {config.api_key_env}")
    params = LiteLlmParamsBase(
        provider=config.provider,
        model_slug=config.model_slug,
        api_key=api_key,
        max_tokens=config.max_tokens,
    )
    request = LiteLlmRequest(
        messages=[
            LiteLlmMessage(
                role="system",
                content=(
                    "You produce strict JSON safe law-DSL priors for SimFutures-LP. "
                    "Never return markdown, Python code, or explanatory prose."
                ),
            ),
            LiteLlmMessage(role="user", content=prior_prompt.prompt),
        ],
        params=params,
    )
    response = request()
    text = _response_text(response)
    return templates_from_llm_json(text, state_dim=state_dim, action_dim=action_dim), text


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


def _response_text(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError("LLM response does not contain choices")
    choice = choices[0]
    message = getattr(choice, "message", None)
    if isinstance(message, dict):
        content = message.get("content")
    else:
        content = getattr(message, "content", None)
    if content is None:
        content = getattr(choice, "text", None)
    if not isinstance(content, str) or not content.strip():
        raise ValueError("LLM response does not contain text content")
    return content.strip()


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
                "law_type": item.law_type,
                "law_gain": item.law_gain,
                "scale": item.scale,
                "prior_mean": item.prior_mean,
                "prior_std": item.prior_std,
                "confidence": item.prior_confidence,
                "timescale": item.timescale,
                "reward_relevance": item.reward_relevance,
                "description": item.description,
            }
            for item in prompt.fallback_templates
        ],
    }
