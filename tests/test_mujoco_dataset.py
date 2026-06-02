from __future__ import annotations

import numpy as np
import torch

from onelife.mujoco_dataset import MuJoCoTransitions


def test_mujoco_transitions_roundtrip_npz(tmp_path):
    transitions = MuJoCoTransitions(
        states=np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32),
        actions=np.array([[0.5], [1.5]], dtype=np.float32),
        next_states=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    )
    path = tmp_path / "mujoco_transitions.npz"

    transitions.save_npz(path)
    loaded = MuJoCoTransitions.load_npz(path)

    assert loaded.num_steps == 2
    assert loaded.state_dim == 2
    assert loaded.action_dim == 1
    assert np.array_equal(loaded.states, transitions.states)
    assert np.array_equal(loaded.actions, transitions.actions)
    assert np.array_equal(loaded.next_states, transitions.next_states)


def test_mujoco_transitions_iter_torch_batches():
    transitions = MuJoCoTransitions(
        states=np.array([[0.0], [1.0], [2.0]], dtype=np.float32),
        actions=np.array([[10.0], [11.0], [12.0]], dtype=np.float32),
        next_states=np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
    )

    batches = list(transitions.iter_torch_batches(batch_size=2, shuffle=False))

    assert len(batches) == 2
    first_states, first_actions, first_next_states = batches[0]
    assert torch.allclose(first_states[:, 0], torch.tensor([0.0, 1.0]))
    assert torch.allclose(first_actions[:, 0], torch.tensor([10.0, 11.0]))
    assert torch.allclose(first_next_states[:, 0], torch.tensor([1.0, 2.0]))
