from __future__ import annotations

import numpy as np

from scripts.run_mujoco_experiment import (
    mujoco_planning_reward_proxy,
    one_step_r2_metrics,
    r2_global,
    r2_uniform,
    rollout_delta_r2_metrics,
    score_action_sequences,
)


def test_r2_uniform_and_global_are_one_for_perfect_predictions():
    targets = np.array([[0.0, 1.0], [1.0, 3.0], [2.0, 5.0]], dtype=np.float32)

    assert r2_uniform(targets, targets) == 1.0
    assert r2_global(targets, targets) == 1.0


def test_one_step_delta_r2_uses_state_delta():
    states = np.array([[10.0], [20.0], [30.0]], dtype=np.float32)
    next_states = np.array([[11.0], [22.0], [33.0]], dtype=np.float32)
    predictions = np.array([[11.0], [22.0], [33.0]], dtype=np.float32)

    metrics = one_step_r2_metrics(states, predictions, next_states)

    assert metrics["one_step_delta_r2_uniform"] == 1.0
    assert metrics["one_step_next_state_r2_uniform"] == 1.0


def test_rollout_delta_r2_uses_cumulative_delta():
    starts = {
        10: [
            np.array([10.0], dtype=np.float32),
            np.array([20.0], dtype=np.float32),
            np.array([30.0], dtype=np.float32),
        ]
    }
    targets = {
        10: [
            np.array([11.0], dtype=np.float32),
            np.array([22.0], dtype=np.float32),
            np.array([33.0], dtype=np.float32),
        ]
    }
    predictions = {
        10: [
            np.array([11.0], dtype=np.float32),
            np.array([22.0], dtype=np.float32),
            np.array([33.0], dtype=np.float32),
        ]
    }

    metrics = rollout_delta_r2_metrics(starts, predictions, targets)

    assert metrics["open_loop_delta_r2_uniform_h10"] == 1.0


def test_reward_proxy_prefers_forward_velocity_for_halfcheetah():
    slow = np.zeros(17, dtype=np.float32)
    fast = np.zeros(17, dtype=np.float32)
    fast[8] = 2.0
    action = np.zeros(6, dtype=np.float32)

    assert mujoco_planning_reward_proxy("HalfCheetah-v5", fast, action) > (
        mujoco_planning_reward_proxy("HalfCheetah-v5", slow, action)
    )


def test_reward_proxy_prefers_forward_velocity_for_swimmer():
    slow = np.zeros(8, dtype=np.float32)
    fast = np.zeros(8, dtype=np.float32)
    fast[3] = 1.0
    action = np.zeros(2, dtype=np.float32)

    assert mujoco_planning_reward_proxy("Swimmer-v5", fast, action) > (
        mujoco_planning_reward_proxy("Swimmer-v5", slow, action)
    )


def test_score_action_sequences_rolls_model_forward():
    def predict_next(state, action):
        return np.asarray(state, dtype=np.float32) + np.asarray(action, dtype=np.float32)

    def observation(state):
        return np.asarray(state, dtype=np.float32)

    start = np.array([0.0] * 17, dtype=np.float32)
    action_sequences = np.zeros((2, 2, 17), dtype=np.float32)
    action_sequences[1, :, 8] = 1.0

    scores = score_action_sequences(
        start,
        action_sequences,
        "HalfCheetah-v5",
        predict_next,
        observation,
    )

    assert scores[1] > scores[0]


def test_score_action_sequences_can_apply_planner_risk_penalty():
    class RiskyState:
        def __init__(self, observation: np.ndarray, risk: float) -> None:
            self.observation = observation
            self.risk = risk

    def predict_next(state, action):
        base = state.observation if isinstance(state, RiskyState) else state
        return RiskyState(
            np.asarray(base, dtype=np.float32) + np.asarray(action, dtype=np.float32),
            risk=10.0,
        )

    def observation(state):
        return state.observation if isinstance(state, RiskyState) else state

    start = np.zeros(17, dtype=np.float32)
    action_sequences = np.zeros((1, 1, 17), dtype=np.float32)
    action_sequences[0, 0, 8] = 1.0

    plain = score_action_sequences(
        start,
        action_sequences,
        "HalfCheetah-v5",
        predict_next,
        observation,
        penalize_risk=False,
    )
    guarded = score_action_sequences(
        start,
        action_sequences,
        "HalfCheetah-v5",
        predict_next,
        observation,
        penalize_risk=True,
    )

    assert plain[0] > guarded[0]
