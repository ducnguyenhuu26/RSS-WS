from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterator

import numpy as np
import torch

from onelife.mujoco_dataset import MuJoCoTransitions
from .templates import MechanismTemplate


@dataclass
class DUCBatch:
    indices: torch.Tensor
    states: torch.Tensor
    actions: torch.Tensor
    next_states: torch.Tensor
    history_states: torch.Tensor
    history_actions: torch.Tensor
    contexts: torch.Tensor | None = None
    rewards: torch.Tensor | None = None


def align_contexts_to_templates(
    transitions: MuJoCoTransitions,
    templates: tuple[MechanismTemplate, ...],
) -> MuJoCoTransitions:
    if transitions.contexts is None:
        return transitions
    if transitions.context_names == tuple(template.name for template in templates):
        return transitions
    name_to_index = {name: index for index, name in enumerate(transitions.context_names)}
    aligned = np.zeros((transitions.num_steps, len(templates)), dtype=np.float32)
    for column, template in enumerate(templates):
        if template.name in name_to_index:
            aligned[:, column] = transitions.contexts[:, name_to_index[template.name]]
        elif template.name == "actuation":
            aligned[:, column] = 1.0
    return MuJoCoTransitions(
        states=transitions.states,
        actions=transitions.actions,
        next_states=transitions.next_states,
        rewards=transitions.rewards,
        dones=transitions.dones,
        contexts=aligned,
        context_names=tuple(template.name for template in templates),
    )


def _history_for_indices(
    values: np.ndarray,
    indices: np.ndarray,
    history_length: int,
) -> np.ndarray:
    history = np.zeros((len(indices), history_length, values.shape[1]), dtype=np.float32)
    for row, index in enumerate(indices):
        start = max(0, int(index) - history_length + 1)
        window = values[start : int(index) + 1]
        if len(window) < history_length:
            pad = np.repeat(window[:1], history_length - len(window), axis=0)
            window = np.concatenate([pad, window], axis=0)
        history[row] = window[-history_length:]
    return history


def iter_duc_batches(
    transitions: MuJoCoTransitions,
    batch_size: int,
    history_length: int,
    shuffle: bool = True,
    seed: int = 0,
    device: torch.device | str | None = None,
) -> Iterator[DUCBatch]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if history_length <= 0:
        raise ValueError("history_length must be positive")
    rng = np.random.default_rng(seed)
    indices = np.arange(transitions.num_steps)
    if shuffle:
        rng.shuffle(indices)
    for start in range(0, transitions.num_steps, batch_size):
        batch_indices = indices[start : start + batch_size]
        history_states = _history_for_indices(transitions.states, batch_indices, history_length)
        history_actions = _history_for_indices(transitions.actions, batch_indices, history_length)
        contexts = None
        if transitions.contexts is not None:
            contexts = torch.tensor(
                transitions.contexts[batch_indices],
                dtype=torch.float32,
                device=device,
            )
        rewards = None
        if transitions.rewards is not None:
            rewards = torch.tensor(
                transitions.rewards[batch_indices],
                dtype=torch.float32,
                device=device,
            )
        yield DUCBatch(
            indices=torch.tensor(batch_indices, dtype=torch.long, device=device),
            states=torch.tensor(transitions.states[batch_indices], dtype=torch.float32, device=device),
            actions=torch.tensor(transitions.actions[batch_indices], dtype=torch.float32, device=device),
            next_states=torch.tensor(transitions.next_states[batch_indices], dtype=torch.float32, device=device),
            history_states=torch.tensor(history_states, dtype=torch.float32, device=device),
            history_actions=torch.tensor(history_actions, dtype=torch.float32, device=device),
            contexts=contexts,
            rewards=rewards,
        )
