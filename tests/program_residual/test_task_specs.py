from __future__ import annotations

import torch

from onelife.program_residual import LLMLawSynthesisConfig, TransitionBatch
from onelife.program_residual.llm_synthesizer import build_mujoco_law_synthesis_prompt
from onelife.program_residual.task_specs import get_mujoco_task_spec


def test_reacher_task_spec_exposes_semantic_dimensions():
    spec = get_mujoco_task_spec("Reacher-v5", state_dim=10, action_dim=2)

    assert spec.state_dimensions[4].name == "target_x"
    assert spec.state_dimensions[8].kind == "geometry"
    assert spec.action_dimensions[0].name == "shoulder_torque"


def test_inverted_double_pendulum_task_spec_matches_v5_observation_size():
    spec = get_mujoco_task_spec("InvertedDoublePendulum-v5", state_dim=9, action_dim=1)

    assert len(spec.state_dimensions) == 9
    assert spec.state_dimensions[8].kind == "constraint"
    assert spec.state_dimensions[8].name == "constraint_force_x"


def test_mujoco_prompt_contains_task_conditioned_semantics():
    batch = TransitionBatch(
        states=torch.zeros(2, 10),
        actions=torch.zeros(2, 2),
        next_states=torch.zeros(2, 10),
    )

    prompt = build_mujoco_law_synthesis_prompt(
        batch=batch,
        state_dim=10,
        action_dim=2,
        config=LLMLawSynthesisConfig(env_id="Reacher-v5", sample_count=1),
    )

    assert "TASK-CONDITIONED MUJOCO SEMANTICS" in prompt
    assert "MUJOCO COORDINATE GROUPS" in prompt
    assert "qpos / generalized position-like coordinates" in prompt
    assert "qvel / generalized velocity coordinates" in prompt
    assert "ctrl / actuator force-torque inputs" in prompt
    assert "state[4] target_x" in prompt
    assert "state[0] cos_joint0_angle" in prompt
    assert "action[0] shoulder_torque" in prompt
    assert "Probabilistic symbolic interpretation" in prompt
    assert "Leader/follower/DAG design rule" in prompt
    assert "concept leaders" in prompt
    assert "qvel velocity coordinates" in prompt
    assert "ctrl torque/force inputs" in prompt
    assert "What to infer:" in prompt
    assert "flat MuJoCo vectors (state, action)" in prompt
    assert "Access coordinates only as `state[i]`" in prompt
    assert "Semantic labels such as qpos, qvel" in prompt
    assert "They are not Python variables, not classes" in prompt
    assert "Never reference invented symbols such as QposCartPosition" in prompt
    assert "law_name` string only" in prompt
    assert "There is no env object, simulator object" in prompt
    assert "Prediction-value contract" in prompt
    assert "predicted next-state values" in prompt
