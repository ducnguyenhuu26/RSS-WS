from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import cloudpickle
import numpy as np
import numpy.typing as npt
import torch

from onelife.mujoco_dataset import MuJoCoTransitions
from onelife.our_method.core import SymbolicTransition as LawSymbolicTransition
from onelife.our_method.core import WeightedLaw
from onelife.our_method.world_modeling import (
    LawMixture,
    combine_active_expert_predictions_for_attr,
)
from onelife.poe_world.core import (
    DiscreteDistribution,
    ObservableId,
    SymbolicTransition as PoESymbolicTransition,
    WeightedExpert,
)
from onelife.poe_world.weight_fitter import combine_expert_predictions_for_attr
from onelife.poe_world.world_model import PoEWorldModel


MuJoCoBinValue: TypeAlias = int | DiscreteDistribution


@dataclass
class BinnedMuJoCoState:
    bins: tuple[MuJoCoBinValue, ...]

    def observed_bins(self) -> tuple[int, ...]:
        observed = []
        for value in self.bins:
            if isinstance(value, DiscreteDistribution):
                raise ValueError("state contains predictions, not observed bins")
            observed.append(int(value))
        return tuple(observed)


@dataclass(frozen=True)
class BinnedMuJoCoAction:
    bins: tuple[int, ...]


@dataclass(frozen=True)
class MuJoCoDiscretizer:
    state_edges: tuple[npt.NDArray[np.float32], ...]
    action_edges: tuple[npt.NDArray[np.float32], ...]

    @classmethod
    def fit(
        cls,
        transitions: MuJoCoTransitions,
        state_bins: int = 21,
        action_bins: int = 11,
    ) -> "MuJoCoDiscretizer":
        if state_bins <= 0:
            raise ValueError("state_bins must be positive")
        if action_bins <= 0:
            raise ValueError("action_bins must be positive")
        observed_states = np.concatenate(
            [transitions.states, transitions.next_states],
            axis=0,
        )
        return cls(
            state_edges=_fit_edges(observed_states, state_bins),
            action_edges=_fit_edges(transitions.actions, action_bins),
        )

    @property
    def state_dim(self) -> int:
        return len(self.state_edges)

    @property
    def action_dim(self) -> int:
        return len(self.action_edges)

    @property
    def num_state_bins(self) -> tuple[int, ...]:
        return tuple(len(edges) + 1 for edges in self.state_edges)

    @property
    def num_action_bins(self) -> tuple[int, ...]:
        return tuple(len(edges) + 1 for edges in self.action_edges)

    def state_support(self, dim: int) -> npt.NDArray[np.int32]:
        return np.arange(self.num_state_bins[dim], dtype=np.int32)

    def digitize_state(self, state: npt.ArrayLike) -> BinnedMuJoCoState:
        return BinnedMuJoCoState(_digitize_vector(state, self.state_edges))

    def digitize_action(self, action: npt.ArrayLike) -> BinnedMuJoCoAction:
        return BinnedMuJoCoAction(_digitize_vector(action, self.action_edges))


@dataclass(frozen=True)
class MuJoCoBinnedObservableExtractor:
    num_state_bins: tuple[int, ...]
    noise_logscore: float = -10.0

    @classmethod
    def from_discretizer(
        cls,
        discretizer: MuJoCoDiscretizer,
        noise_logscore: float = -10.0,
    ) -> "MuJoCoBinnedObservableExtractor":
        return cls(
            num_state_bins=discretizer.num_state_bins,
            noise_logscore=noise_logscore,
        )

    def extract_attribute_predictions(
        self,
        state: BinnedMuJoCoState,
    ) -> dict[ObservableId, DiscreteDistribution]:
        predictions: dict[ObservableId, DiscreteDistribution] = {}
        for dim, value in enumerate(state.bins):
            if isinstance(value, DiscreteDistribution):
                predictions[_state_observable_id(dim)] = value.expand_support(
                    self._support(dim),
                    noise_logscore=self.noise_logscore,
                )
        return predictions

    def get_observed_outcomes(
        self,
        state: BinnedMuJoCoState,
    ) -> dict[ObservableId, int]:
        return {
            _state_observable_id(dim): value
            for dim, value in enumerate(state.observed_bins())
        }

    def apply_expert_predictions(
        self,
        new_state: BinnedMuJoCoState,
        expert_predictions: Mapping[
            ObservableId,
            Sequence[DiscreteDistribution] | Mapping[int, DiscreteDistribution],
        ],
        weights: torch.Tensor,
    ) -> BinnedMuJoCoState:
        bins = list(new_state.bins)
        for dim in range(len(self.num_state_bins)):
            attr = _state_observable_id(dim)
            if attr not in expert_predictions:
                continue
            raw_predictions = expert_predictions[attr]
            if isinstance(raw_predictions, Mapping):
                combined = combine_active_expert_predictions_for_attr(
                    raw_predictions,
                    weights,
                )
            else:
                combined = combine_expert_predictions_for_attr(
                    raw_predictions,
                    weights,
                )
            bins[dim] = int(combined.sample())
        new_state.bins = tuple(bins)
        return new_state

    def _support(self, dim: int) -> npt.NDArray[np.int32]:
        return np.arange(self.num_state_bins[dim], dtype=np.int32)


class IdentityMuJoCoLaw:
    def __init__(self, state_dim: int):
        self.state_dim = state_dim

    def precondition(
        self,
        current_state: BinnedMuJoCoState,
        action: BinnedMuJoCoAction,
    ) -> bool:
        return True

    def effect(
        self,
        current_state: BinnedMuJoCoState,
        action: BinnedMuJoCoAction,
    ) -> None:
        current_state.bins = tuple(
            DiscreteDistribution(support=[int(value)])
            for value in current_state.observed_bins()
        )

    @property
    def __source_code__(self) -> str:
        return "IdentityMuJoCoLaw: predicts each discretized state bin stays fixed."

    @property
    def __name__(self) -> str:
        return "IdentityMuJoCoLaw"

    def save(self, path: str | Path) -> None:
        _save_cloudpickle(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "IdentityMuJoCoLaw":
        return _load_cloudpickle(path, cls)


class ActionDeltaMuJoCoLaw:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_state_bins: int,
        num_action_bins: int,
        delta: int = 1,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_state_bins = num_state_bins
        self.num_action_bins = num_action_bins
        self.delta = delta

    def precondition(
        self,
        current_state: BinnedMuJoCoState,
        action: BinnedMuJoCoAction,
    ) -> bool:
        action_bin = action.bins[self.action_dim]
        return action_bin != self.num_action_bins // 2

    def effect(
        self,
        current_state: BinnedMuJoCoState,
        action: BinnedMuJoCoAction,
    ) -> None:
        bins = list(current_state.observed_bins())
        direction = 1 if action.bins[self.action_dim] > self.num_action_bins // 2 else -1
        next_bin = int(
            np.clip(
                bins[self.state_dim] + direction * self.delta,
                0,
                self.num_state_bins - 1,
            )
        )
        bins[self.state_dim] = DiscreteDistribution(support=[next_bin])
        current_state.bins = tuple(bins)

    @property
    def __source_code__(self) -> str:
        return (
            "ActionDeltaMuJoCoLaw: moves one discretized state dimension by the "
            "sign of one discretized action dimension."
        )

    @property
    def __name__(self) -> str:
        return f"ActionDeltaMuJoCoLaw_s{self.state_dim}_a{self.action_dim}"

    def save(self, path: str | Path) -> None:
        _save_cloudpickle(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "ActionDeltaMuJoCoLaw":
        return _load_cloudpickle(path, cls)


class LawAsPoEExpert:
    def __init__(self, law: IdentityMuJoCoLaw | ActionDeltaMuJoCoLaw):
        self.law = law

    def __call__(
        self,
        current_state: BinnedMuJoCoState,
        action: BinnedMuJoCoAction,
    ) -> None:
        if self.law.precondition(current_state, action):
            self.law.effect(current_state, action)

    @property
    def __source_code__(self) -> str:
        return self.law.__source_code__

    @property
    def __name__(self) -> str:
        return self.law.__name__

    def save(self, path: str | Path) -> None:
        _save_cloudpickle(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "LawAsPoEExpert":
        return _load_cloudpickle(path, cls)


def to_poe_symbolic_transitions(
    transitions: MuJoCoTransitions,
    discretizer: MuJoCoDiscretizer,
) -> list[PoESymbolicTransition[BinnedMuJoCoState]]:
    return [
        PoESymbolicTransition(
            prev_metadata=discretizer.digitize_state(state),
            action=discretizer.digitize_action(action),
            next_metadata=discretizer.digitize_state(next_state),
        )
        for state, action, next_state in zip(
            transitions.states,
            transitions.actions,
            transitions.next_states,
        )
    ]


def to_law_symbolic_transitions(
    transitions: MuJoCoTransitions,
    discretizer: MuJoCoDiscretizer,
) -> list[LawSymbolicTransition[BinnedMuJoCoState]]:
    return [
        LawSymbolicTransition(
            prev_state=discretizer.digitize_state(state),
            action=discretizer.digitize_action(action),
            next_state=discretizer.digitize_state(next_state),
        )
        for state, action, next_state in zip(
            transitions.states,
            transitions.actions,
            transitions.next_states,
        )
    ]


def make_default_mujoco_laws(
    discretizer: MuJoCoDiscretizer,
) -> list[IdentityMuJoCoLaw | ActionDeltaMuJoCoLaw]:
    laws: list[IdentityMuJoCoLaw | ActionDeltaMuJoCoLaw] = [
        IdentityMuJoCoLaw(discretizer.state_dim)
    ]
    for dim in range(min(discretizer.state_dim, discretizer.action_dim)):
        laws.append(
            ActionDeltaMuJoCoLaw(
                state_dim=dim,
                action_dim=dim,
                num_state_bins=discretizer.num_state_bins[dim],
                num_action_bins=discretizer.num_action_bins[dim],
            )
        )
    return laws


def make_poe_mujoco_baseline(
    discretizer: MuJoCoDiscretizer,
    laws: Sequence[IdentityMuJoCoLaw | ActionDeltaMuJoCoLaw] | None = None,
    weight: float = 1.0,
) -> PoEWorldModel[BinnedMuJoCoState, BinnedMuJoCoAction]:
    laws = list(laws) if laws is not None else make_default_mujoco_laws(discretizer)
    return PoEWorldModel(
        observable_extractor=MuJoCoBinnedObservableExtractor.from_discretizer(
            discretizer
        ),
        weighted_experts=[
            WeightedExpert(
                expert_function=LawAsPoEExpert(law),
                weight=weight,
                is_fitted=True,
            )
            for law in laws
        ],
    )


def make_onelife_mujoco_law_mixture(
    discretizer: MuJoCoDiscretizer,
    laws: Sequence[IdentityMuJoCoLaw | ActionDeltaMuJoCoLaw] | None = None,
    weight: float = 1.0,
) -> LawMixture[BinnedMuJoCoState, BinnedMuJoCoAction]:
    laws = list(laws) if laws is not None else make_default_mujoco_laws(discretizer)
    return LawMixture(
        observable_extractor=MuJoCoBinnedObservableExtractor.from_discretizer(
            discretizer
        ),
        weighted_laws=[
            WeightedLaw(
                law=law,
                weight=weight,
                is_fitted=True,
            )
            for law in laws
        ],
    )


def _fit_edges(
    values: npt.NDArray[np.float32],
    num_bins: int,
) -> tuple[npt.NDArray[np.float32], ...]:
    edges = []
    for dim in range(values.shape[1]):
        column = values[:, dim]
        min_value = float(np.min(column))
        max_value = float(np.max(column))
        if min_value == max_value:
            min_value -= 0.5
            max_value += 0.5
        dim_edges = np.linspace(
            min_value,
            max_value,
            num_bins + 1,
            dtype=np.float32,
        )[1:-1]
        edges.append(dim_edges)
    return tuple(edges)


def _digitize_vector(
    values: npt.ArrayLike,
    edges: Sequence[npt.NDArray[np.float32]],
) -> tuple[int, ...]:
    array = np.asarray(values, dtype=np.float32)
    if array.shape[-1] != len(edges):
        raise ValueError("vector dimension does not match fitted discretizer")
    return tuple(
        int(np.digitize(float(array[dim]), edges[dim]))
        for dim in range(len(edges))
    )


def _state_observable_id(dim: int) -> ObservableId:
    return ObservableId(f"state_{dim}")


def _save_cloudpickle(instance: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        cloudpickle.dump(instance, file)


def _load_cloudpickle(path: str | Path, expected_type: type[Any]) -> Any:
    with Path(path).open("rb") as file:
        instance = cloudpickle.load(file)
    if not isinstance(instance, expected_type):
        raise TypeError(
            f"File '{path}' contained {type(instance).__name__}, "
            f"not {expected_type.__name__}."
        )
    return instance
