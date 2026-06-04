from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from onelife.program_residual import (
    KinematicPositionLaw,
    ProgramResidualTrainerConfig,
    ProgramResidualWorldModel,
    ResidualMLP,
    SymbolicProgram,
    TransitionBatch,
    collect_transitions_from_env,
    compute_program_residual_loss,
    make_optimizer,
    train_step,
)


class ConstantResidual(nn.Module):
    def __init__(self, residual: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("residual", residual)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        program_next_states: torch.Tensor,
        unknown_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.residual.expand_as(states)


class FixedGate(nn.Module):
    def __init__(self, gate: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("gate", gate)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        symbolic_delta: torch.Tensor,
        confidence: torch.Tensor,
        unknown_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.gate.expand_as(states)


def test_symbolic_program_marks_uncovered_dimensions_unknown():
    program = SymbolicProgram(
        state_dim=2,
        laws=[
            KinematicPositionLaw(
                position_indices=[0],
                velocity_indices=[1],
                dt=0.1,
                confidence=2.0,
            )
        ],
    )

    output = program(
        torch.tensor([1.0, 3.0]),
        torch.tensor([0.0]),
    )

    assert torch.allclose(output.next_state, torch.tensor([1.3, 3.0]))
    assert torch.allclose(output.confidence, torch.tensor([2.0, 0.0]))
    assert torch.allclose(output.unknown_mask, torch.tensor([0.0, 1.0]))
    assert output.active_laws == (("KinematicPositionLaw",),)


def test_program_residual_model_applies_residual_only_to_unknown_dimensions():
    program = SymbolicProgram(
        state_dim=2,
        laws=[
            KinematicPositionLaw(
                position_indices=[0],
                velocity_indices=[1],
                dt=0.1,
                confidence=1.0,
            )
        ],
    )
    model = ProgramResidualWorldModel(
        state_dim=2,
        action_dim=1,
        program=program,
        residual_model=ConstantResidual(torch.tensor([10.0, -1.0])),
        apply_unknown_mask=True,
    )

    output = model(torch.tensor([1.0, 3.0]), torch.tensor([0.0]))

    assert torch.allclose(output.program_next_state, torch.tensor([1.3, 3.0]))
    assert torch.allclose(output.applied_residual, torch.tensor([0.0, -1.0]))
    assert torch.allclose(output.prediction, torch.tensor([1.3, 2.0]))


def test_program_residual_model_default_allows_residual_to_correct_known_dimensions():
    program = SymbolicProgram(
        state_dim=2,
        laws=[
            KinematicPositionLaw(
                position_indices=[0],
                velocity_indices=[1],
                dt=0.1,
                confidence=1.0,
            )
        ],
    )
    model = ProgramResidualWorldModel(
        state_dim=2,
        action_dim=1,
        program=program,
        residual_model=ConstantResidual(torch.tensor([10.0, -1.0])),
    )

    output = model(torch.tensor([1.0, 3.0]), torch.tensor([0.0]))

    assert torch.allclose(output.program_next_state, torch.tensor([1.3, 3.0]))
    assert torch.allclose(output.applied_residual, torch.tensor([10.0, -1.0]))
    assert torch.allclose(output.prediction, torch.tensor([11.3, 2.0]))


def test_gated_model_uses_symbolic_only_for_known_dimensions():
    program = SymbolicProgram(
        state_dim=2,
        laws=[
            KinematicPositionLaw(
                position_indices=[0],
                velocity_indices=[1],
                dt=0.1,
                confidence=1.0,
            )
        ],
    )
    model = ProgramResidualWorldModel(
        state_dim=2,
        action_dim=1,
        program=program,
        residual_model=ConstantResidual(torch.tensor([10.0, -1.0])),
        gate_model=FixedGate(torch.tensor([1.0, 1.0])),
    )

    output = model(torch.tensor([1.0, 3.0]), torch.tensor([0.0]))

    assert torch.allclose(output.symbolic_gate, torch.tensor([1.0, 0.0]))
    assert torch.allclose(output.prediction, torch.tensor([1.3, 2.0]))


def test_training_step_updates_residual_model_parameters():
    torch.manual_seed(0)
    program = SymbolicProgram(state_dim=2, laws=[])
    residual = ResidualMLP(
        state_dim=2,
        action_dim=1,
        hidden_sizes=(8,),
        zero_init_output=False,
    )
    model = ProgramResidualWorldModel(
        state_dim=2,
        action_dim=1,
        program=program,
        residual_model=residual,
    )
    batch = TransitionBatch(
        states=torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
        actions=torch.tensor([[1.0], [1.0]]),
        next_states=torch.tensor([[0.5, -0.5], [1.5, 0.5]]),
    )
    config = ProgramResidualTrainerConfig(
        learning_rate=1e-2,
        residual_l2_weight=0.0,
        max_grad_norm=None,
    )
    optimizer = make_optimizer(model, config)

    before = model(batch.states, batch.actions).prediction.detach().clone()
    metrics = train_step(model, optimizer, batch, config)
    after = model(batch.states, batch.actions).prediction.detach().clone()

    assert metrics.loss > 0
    assert not torch.allclose(before, after)


def test_loss_penalizes_applied_residual():
    output = type(
        "Output",
        (),
        {
            "prediction": torch.tensor([[1.0, 0.0]]),
            "applied_residual": torch.tensor([[2.0, 0.0]]),
            "unknown_mask": torch.tensor([[1.0, 0.0]]),
        },
    )()

    loss, metrics = compute_program_residual_loss(
        output=output,
        target_next_states=torch.tensor([[0.0, 0.0]]),
        residual_l2_weight=0.5,
    )

    assert torch.isclose(loss, torch.tensor(1.5))
    assert torch.isclose(metrics["prediction_loss"], torch.tensor(0.5))
    assert torch.isclose(metrics["residual_l2"], torch.tensor(2.0))


class FakeActionSpace:
    def __init__(self) -> None:
        self.count = 0

    def seed(self, seed: int) -> None:
        self.count = seed

    def sample(self) -> np.ndarray:
        self.count += 1
        return np.array([float(self.count)], dtype=np.float32)


class FakeEnv:
    def __init__(self) -> None:
        self.action_space = FakeActionSpace()
        self.state = 0.0

    def reset(self, seed: int | None = None):
        self.state = 0.0
        return np.array([self.state], dtype=np.float32), {}

    def step(self, action: np.ndarray):
        previous = self.state
        self.state = previous + float(action[0])
        return np.array([self.state], dtype=np.float32), 0.0, False, False, {}


def test_collect_transitions_from_env():
    batch = collect_transitions_from_env(FakeEnv(), num_steps=3, seed=0)

    assert batch.states.shape == (3, 1)
    assert batch.actions.shape == (3, 1)
    assert batch.next_states.shape == (3, 1)
    assert torch.allclose(batch.states[:, 0], torch.tensor([0.0, 1.0, 3.0]))
    assert torch.allclose(batch.actions[:, 0], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(batch.next_states[:, 0], torch.tensor([1.0, 3.0, 6.0]))
