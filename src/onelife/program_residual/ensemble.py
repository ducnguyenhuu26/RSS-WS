from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from onelife.mujoco_dataset import MuJoCoTransitions

from .core import ModelOutput, TransitionBatch
from .model import ProgramResidualWorldModel
from .program import SymbolicProgram
from .residual import ResidualMLP
from .trainer import ProgramResidualTrainerConfig, TrainingMetrics, fit_supervised


@dataclass(frozen=True)
class NeuralEnsembleConfig:
    ensemble_size: int = 5
    bootstrap: bool = True


class NeuralEnsembleWorldModel(nn.Module):
    """PETS-style ensemble of neural dynamics models with a shared MPC interface."""

    def __init__(self, members: Sequence[ProgramResidualWorldModel]) -> None:
        super().__init__()
        if not members:
            raise ValueError("ensemble must contain at least one member")
        state_dim = members[0].state_dim
        action_dim = members[0].action_dim
        if any(member.state_dim != state_dim for member in members):
            raise ValueError("all ensemble members must share state_dim")
        if any(member.action_dim != action_dim for member in members):
            raise ValueError("all ensemble members must share action_dim")
        self.members = nn.ModuleList(members)
        self.state_dim = state_dim
        self.action_dim = action_dim

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> ModelOutput:
        squeeze = states.ndim == 1
        states_batched = states.unsqueeze(0) if squeeze else states
        predictions = torch.stack(
            [member(states, actions).prediction for member in self.members],
            dim=0,
        )
        mean_prediction = predictions.mean(dim=0)
        variance = predictions.var(dim=0, unbiased=False)
        if squeeze:
            reference_states = states
            batch_size = 1
        else:
            reference_states = states_batched
            batch_size = int(states_batched.shape[0])
        residual = mean_prediction - reference_states
        confidence = 1.0 / (1.0 + variance)
        unknown_mask = torch.ones_like(mean_prediction)
        return ModelOutput(
            prediction=mean_prediction,
            program_next_state=reference_states,
            residual=residual,
            applied_residual=residual,
            confidence=confidence,
            unknown_mask=unknown_mask,
            active_laws=tuple(() for _ in range(batch_size)),
            ensemble_variance=variance,
        )

    @torch.no_grad()
    def predict_next_state(
        self,
        state: torch.Tensor | np.ndarray,
        action: torch.Tensor | np.ndarray,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        input_device = state.device if isinstance(state, torch.Tensor) else torch.device("cpu")
        model_device = _module_device(self)
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=model_device)
        action_tensor = torch.as_tensor(action, dtype=torch.float32, device=model_device)
        output = self(state_tensor, action_tensor)
        if was_training:
            self.train()
        return output.prediction.to(input_device)


def build_neural_ensemble_world_model(
    state_dim: int,
    action_dim: int,
    hidden_sizes: Sequence[int],
    ensemble_size: int = 5,
) -> NeuralEnsembleWorldModel:
    if ensemble_size <= 0:
        raise ValueError("ensemble_size must be positive")
    members = [
        ProgramResidualWorldModel(
            state_dim=state_dim,
            action_dim=action_dim,
            program=SymbolicProgram(state_dim=state_dim, laws=[]),
            residual_model=ResidualMLP(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_sizes=tuple(int(size) for size in hidden_sizes),
            ),
        )
        for _ in range(ensemble_size)
    ]
    return NeuralEnsembleWorldModel(members)


def fit_neural_ensemble(
    model: NeuralEnsembleWorldModel,
    dataset: MuJoCoTransitions,
    batch_size: int,
    config: ProgramResidualTrainerConfig,
    num_epochs: int,
    seed: int,
    device: torch.device | str,
    bootstrap: bool = True,
) -> list[list[TrainingMetrics]]:
    histories: list[list[TrainingMetrics]] = []
    for member_index, member in enumerate(model.members):
        batches = [
            batch.to(device)
            for batch in _bootstrap_batches(
                dataset=dataset,
                batch_size=batch_size,
                seed=seed + 10_000 * (member_index + 1),
                bootstrap=bootstrap,
            )
        ]
        histories.append(
            fit_supervised(
                model=member,
                batches=batches,
                config=config,
                num_epochs=num_epochs,
            )
        )
    return histories


def _bootstrap_batches(
    dataset: MuJoCoTransitions,
    batch_size: int,
    seed: int,
    bootstrap: bool,
) -> list[TransitionBatch]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    rng = np.random.default_rng(seed)
    if bootstrap:
        indices = rng.integers(0, dataset.num_steps, size=dataset.num_steps)
    else:
        indices = np.arange(dataset.num_steps)
        rng.shuffle(indices)
    batches: list[TransitionBatch] = []
    for start in range(0, dataset.num_steps, batch_size):
        batch_indices = indices[start : start + batch_size]
        batches.append(
            TransitionBatch(
                states=torch.tensor(dataset.states[batch_indices], dtype=torch.float32),
                actions=torch.tensor(dataset.actions[batch_indices], dtype=torch.float32),
                next_states=torch.tensor(
                    dataset.next_states[batch_indices],
                    dtype=torch.float32,
                ),
            )
        )
    return batches


def _module_device(module: nn.Module) -> torch.device:
    parameter = next(module.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(module.buffers(), None)
    if buffer is not None:
        return buffer.device
    return torch.device("cpu")
