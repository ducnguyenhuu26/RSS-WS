from __future__ import annotations

import json

from onelife.duc_wm import (
    build_duc_mujoco_prior_prompt,
    score_prior_portfolio,
    templates_from_llm_json,
)


def test_duc_prior_prompts_are_environment_specific():
    ant = build_duc_mujoco_prior_prompt("Ant-v5", state_dim=27, action_dim=8)
    reacher = build_duc_mujoco_prior_prompt("Reacher-v5", state_dim=10, action_dim=2)

    assert ant.prompt != reacher.prompt
    assert "quadruped" in ant.prompt.lower()
    assert "reaching arm" in reacher.prompt.lower()
    assert "reward-critical" in ant.prompt.lower()
    assert "confidence" in ant.prompt
    assert "Return JSON only" in ant.prompt


def test_templates_from_llm_json_validates_indices():
    payload = {
        "env_id": "FakeEnv",
        "templates": [
            {
                "name": "wind",
                "state_indices": [0],
                "action_indices": [],
                "output_indices": [1],
                "scale": 0.5,
                "prior_std": 0.2,
                "confidence": 0.7,
                "description": "external drift",
            }
        ],
    }

    templates = templates_from_llm_json(
        "```json\n" + json.dumps(payload) + "\n```",
        state_dim=2,
        action_dim=1,
    )

    assert len(templates) == 1
    assert templates[0].name == "wind"


def test_prior_portfolio_score_prefers_structured_action_laws():
    prompt = build_duc_mujoco_prior_prompt("Ant-v5", state_dim=27, action_dim=8)
    structured = prompt.fallback_templates
    weak_payload = {
        "env_id": "Ant-v5",
        "templates": [
            {
                "name": "unknown",
                "state_indices": list(range(27)),
                "action_indices": [],
                "output_indices": list(range(27)),
                "law_type": "learned_residual",
                "law_gain": 0.0,
                "scale": 0.5,
                "prior_std": 0.8,
                "confidence": 0.1,
                "timescale": "unknown",
                "reward_relevance": "",
                "description": "fallback residual",
            }
        ],
    }
    weak = templates_from_llm_json(json.dumps(weak_payload), state_dim=27, action_dim=8)

    assert score_prior_portfolio(structured, state_dim=27, action_dim=8) > score_prior_portfolio(
        weak,
        state_dim=27,
        action_dim=8,
    )
