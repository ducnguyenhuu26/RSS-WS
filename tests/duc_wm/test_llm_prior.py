from __future__ import annotations

import json

from onelife.duc_wm import build_duc_mujoco_prior_prompt, templates_from_llm_json


def test_duc_prior_prompts_are_environment_specific():
    ant = build_duc_mujoco_prior_prompt("Ant-v5", state_dim=27, action_dim=8)
    reacher = build_duc_mujoco_prior_prompt("Reacher-v5", state_dim=10, action_dim=2)

    assert ant.prompt != reacher.prompt
    assert "quadruped" in ant.prompt.lower()
    assert "reaching arm" in reacher.prompt.lower()
    assert "Return JSON only" in ant.prompt


def test_templates_from_llm_json_validates_indices():
    payload = {
        "env_id": "Fake-v0",
        "templates": [
            {
                "name": "wind",
                "state_indices": [0],
                "action_indices": [],
                "output_indices": [1],
                "scale": 0.5,
                "prior_std": 0.2,
                "description": "external drift",
            }
        ],
    }

    templates = templates_from_llm_json(json.dumps(payload), state_dim=2, action_dim=1)

    assert len(templates) == 1
    assert templates[0].name == "wind"
