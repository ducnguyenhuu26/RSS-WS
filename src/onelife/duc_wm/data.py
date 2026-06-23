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


@dataclass
class PreparedDUCData:
    indices: torch.Tensor
    states: torch.Tensor
    actions: torch.Tensor
    next_states: torch.Tensor
    history_states: torch.Tensor
    history_actions: torch.Tensor
    contexts: torch.Tensor | None = None
    rewards: torch.Tensor | None = None

    @property
    def num_steps(self) -> int:
        return int(self.states.shape[0])


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
    dones: np.ndarray | None = None,
) -> np.ndarray:
    history = np.zeros((len(indices), history_length, values.shape[1]), dtype=np.float32)
    for row, index in enumerate(indices):
        start = max(0, int(index) - history_length + 1)
        if dones is not None and int(index) > 0:
            done_positions = np.flatnonzero(dones[start:int(index)])
            if len(done_positions) > 0:
                start = start + int(done_positions[-1]) + 1
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
        history_states = _history_for_indices(
            transitions.states,
            batch_indices,
            history_length,
            dones=transitions.dones,
        )
        history_actions = _history_for_indices(
            transitions.actions,
            batch_indices,
            history_length,
            dones=transitions.dones,
        )
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


def prepare_duc_data(
    transitions: MuJoCoTransitions,
    history_length: int,
    device: torch.device | str | None = None,
) -> PreparedDUCData:
    indices_np = np.arange(transitions.num_steps)
    history_states = _history_for_indices(
        transitions.states,
        indices_np,
        history_length,
        dones=transitions.dones,
    )
    history_actions = _history_for_indices(
        transitions.actions,
        indices_np,
        history_length,
        dones=transitions.dones,
    )
    contexts = None
    if transitions.contexts is not None:
        contexts = torch.tensor(transitions.contexts, dtype=torch.float32, device=device)
    rewards = None
    if transitions.rewards is not None:
        rewards = torch.tensor(transitions.rewards, dtype=torch.float32, device=device)
    return PreparedDUCData(
        indices=torch.tensor(indices_np, dtype=torch.long, device=device),
        states=torch.tensor(transitions.states, dtype=torch.float32, device=device),
        actions=torch.tensor(transitions.actions, dtype=torch.float32, device=device),
        next_states=torch.tensor(transitions.next_states, dtype=torch.float32, device=device),
        history_states=torch.tensor(history_states, dtype=torch.float32, device=device),
        history_actions=torch.tensor(history_actions, dtype=torch.float32, device=device),
        contexts=contexts,
        rewards=rewards,
    )


def iter_prepared_duc_batches(
    prepared: PreparedDUCData,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 0,
) -> Iterator[DUCBatch]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if shuffle:
        generator = torch.Generator(device=prepared.indices.device)
        generator.manual_seed(seed)
        indices = torch.randperm(prepared.num_steps, generator=generator, device=prepared.indices.device)
    else:
        indices = prepared.indices
    for start in range(0, prepared.num_steps, batch_size):
        batch_positions = indices[start : start + batch_size]
        contexts = None
        if prepared.contexts is not None:
            contexts = prepared.contexts.index_select(dim=0, index=batch_positions)
        rewards = None
        if prepared.rewards is not None:
            rewards = prepared.rewards.index_select(dim=0, index=batch_positions)
        yield DUCBatch(
            indices=batch_positions,
            states=prepared.states.index_select(dim=0, index=batch_positions),
            actions=prepared.actions.index_select(dim=0, index=batch_positions),
            next_states=prepared.next_states.index_select(dim=0, index=batch_positions),
            history_states=prepared.history_states.index_select(dim=0, index=batch_positions),
            history_actions=prepared.history_actions.index_select(dim=0, index=batch_positions),
            contexts=contexts,
            rewards=rewards,
        )
