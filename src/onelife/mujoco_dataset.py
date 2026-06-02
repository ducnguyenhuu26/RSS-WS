from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch


@dataclass(frozen=True)
class MuJoCoTransitions:
    """Shared continuous-control transition dataset for MuJoCo-style tasks."""

    states: npt.NDArray[np.float32]
    actions: npt.NDArray[np.float32]
    next_states: npt.NDArray[np.float32]

    def __post_init__(self) -> None:
        if self.states.ndim != 2:
            raise ValueError("states must have shape [num_steps, state_dim]")
        if self.actions.ndim != 2:
            raise ValueError("actions must have shape [num_steps, action_dim]")
        if self.next_states.ndim != 2:
            raise ValueError("next_states must have shape [num_steps, state_dim]")
        if self.states.shape != self.next_states.shape:
            raise ValueError("states and next_states must have matching shape")
        if self.states.shape[0] != self.actions.shape[0]:
            raise ValueError("states and actions must have matching num_steps")

    @property
    def num_steps(self) -> int:
        return int(self.states.shape[0])

    @property
    def state_dim(self) -> int:
        return int(self.states.shape[1])

    @property
    def action_dim(self) -> int:
        return int(self.actions.shape[1])

    def save_npz(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            states=self.states,
            actions=self.actions,
            next_states=self.next_states,
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> "MuJoCoTransitions":
        loaded = np.load(Path(path))
        return cls(
            states=loaded["states"].astype(np.float32),
            actions=loaded["actions"].astype(np.float32),
            next_states=loaded["next_states"].astype(np.float32),
        )

    def to_torch(
        self,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        states = torch.tensor(self.states, dtype=torch.float32, device=device)
        actions = torch.tensor(self.actions, dtype=torch.float32, device=device)
        next_states = torch.tensor(self.next_states, dtype=torch.float32, device=device)
        return states, actions, next_states

    def iter_torch_batches(
        self,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        device: torch.device | str | None = None,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        rng = np.random.default_rng(seed)
        indices = np.arange(self.num_steps)
        if shuffle:
            rng.shuffle(indices)
        for start in range(0, self.num_steps, batch_size):
            batch_indices = indices[start : start + batch_size]
            states = torch.tensor(
                self.states[batch_indices],
                dtype=torch.float32,
                device=device,
            )
            actions = torch.tensor(
                self.actions[batch_indices],
                dtype=torch.float32,
                device=device,
            )
            next_states = torch.tensor(
                self.next_states[batch_indices],
                dtype=torch.float32,
                device=device,
            )
            yield states, actions, next_states
