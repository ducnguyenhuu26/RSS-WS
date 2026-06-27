from __future__ import annotations

import numpy as np

from onelife.duc_wm import build_planner_coverage_stats
from onelife.mujoco_dataset import MuJoCoTransitions


def test_planner_coverage_stats_distance_increases_off_support() -> None:
    states = np.zeros((16, 3), dtype=np.float32)
    actions = np.zeros((16, 2), dtype=np.float32)
    next_states = states.copy()
    transitions = MuJoCoTransitions(
        states=states,
        actions=actions,
        next_states=next_states,
        rewards=np.zeros(16, dtype=np.float32),
        dones=np.zeros(16, dtype=np.bool_),
    )

    stats = build_planner_coverage_stats(transitions)
    in_support = stats.distance(np.zeros(3, dtype=np.float32), np.zeros(2, dtype=np.float32))
    off_support = stats.distance(np.ones(3, dtype=np.float32) * 5.0, np.ones(2, dtype=np.float32) * 5.0)

    assert in_support == 0.0
    assert off_support > in_support
    assert stats.train_p95 == 0.0
