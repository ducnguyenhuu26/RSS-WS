from ...poe_world.core import (
    ObservableId,
    DiscreteDistribution,
)
from ..core import ObservableExtractorProtocol
from ...mixed_fsm_env import State, Action
from ...typing_utils import implements
import numpy as np
import torch
from ..world_modeling import combine_active_expert_predictions_for_attr
from typing import Mapping, TypeAlias


ExpertIndex: TypeAlias = int


class ObservableExtractor:
    def __init__(self):
        self.deterministic_switch_domain = np.array([0, 1])
        self.stochastic_switch_domain = np.array([0, 1])
        self.static_switch_domain = np.array([0, 1])

    def extract_attribute_predictions(
        self, state: State
    ) -> dict[ObservableId, DiscreteDistribution]:
        predictions: dict[ObservableId, DiscreteDistribution] = {}

        if isinstance(state.deterministic_switch, DiscreteDistribution):
            predictions[ObservableId("deterministic_switch")] = (
                state.deterministic_switch.expand_support(
                    self.deterministic_switch_domain
                )
            )

        if isinstance(state.stochastic_switch, DiscreteDistribution):
            predictions[ObservableId("stochastic_switch")] = (
                state.stochastic_switch.expand_support(self.stochastic_switch_domain)
            )

        if isinstance(state.static_switch, DiscreteDistribution):
            predictions[ObservableId("static_switch")] = (
                state.static_switch.expand_support(self.static_switch_domain)
            )

        return predictions

    def get_observed_outcomes(self, state: State) -> dict[ObservableId, int]:
        return {
            ObservableId("deterministic_switch"): state.deterministic_switch,
            ObservableId("stochastic_switch"): state.stochastic_switch,
            ObservableId("static_switch"): state.static_switch,
        }

    @staticmethod
    def apply_expert_predictions(
        new_state: State,
        expert_predictions: Mapping[
            ObservableId, Mapping[ExpertIndex, DiscreteDistribution]
        ],
        weights: torch.Tensor,
    ) -> State:
        if "deterministic_switch" in expert_predictions:
            deterministic_switch_preds = expert_predictions[
                ObservableId("deterministic_switch")
            ]
            combined_dist = combine_active_expert_predictions_for_attr(
                deterministic_switch_preds, weights
            )
            new_state.deterministic_switch = combined_dist.sample()

        if "stochastic_switch" in expert_predictions:
            stochastic_switch_preds = expert_predictions[
                ObservableId("stochastic_switch")
            ]
            combined_dist = combine_active_expert_predictions_for_attr(
                stochastic_switch_preds, weights
            )
            new_state.stochastic_switch = combined_dist.sample()

        if "static_switch" in expert_predictions:
            static_switch_preds = expert_predictions[ObservableId("static_switch")]
            combined_dist = combine_active_expert_predictions_for_attr(
                static_switch_preds, weights
            )
            new_state.static_switch = combined_dist.sample()

        return new_state


implements(ObservableExtractorProtocol[State])(ObservableExtractor)
