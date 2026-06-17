from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True)
class TransitionBatch:
    """A batch of continuous-control transitions."""

    states: torch.Tensor
    actions: torch.Tensor
    next_states: torch.Tensor

    def to(self, device: torch.device | str) -> "TransitionBatch":
        return TransitionBatch(
            states=self.states.to(device),
            actions=self.actions.to(device),
            next_states=self.next_states.to(device),
        )


@dataclass(frozen=True)
class LawPrediction:
    """A continuous law prediction over selected next-state dimensions."""

    indices: torch.Tensor
    values: torch.Tensor
    confidence: torch.Tensor
    law_name: str
    std: torch.Tensor | None = None
    weight: torch.Tensor | None = None


@dataclass(frozen=True)
class ProgramOutput:
    """Output of the executable symbolic program for one or more states."""

    next_state: torch.Tensor
    confidence: torch.Tensor
    unknown_mask: torch.Tensor
    active_laws: tuple[tuple[str, ...], ...]
    variance: torch.Tensor | None = None


@dataclass(frozen=True)
class ModelOutput:
    """Full program-residual model output."""

    prediction: torch.Tensor
    program_next_state: torch.Tensor
    residual: torch.Tensor
    applied_residual: torch.Tensor
    confidence: torch.Tensor
    unknown_mask: torch.Tensor
    active_laws: tuple[tuple[str, ...], ...]
    symbolic_gate: torch.Tensor | None = None
    ensemble_variance: torch.Tensor | None = None
    program_variance: torch.Tensor | None = None


class ContinuousLawProtocol(Protocol):
    """Protocol for executable continuous symbolic laws."""

    def precondition(self, state: torch.Tensor, action: torch.Tensor) -> bool: ...

    def predict(self, state: torch.Tensor, action: torch.Tensor) -> LawPrediction: ...

    @property
    def law_name(self) -> str: ...
