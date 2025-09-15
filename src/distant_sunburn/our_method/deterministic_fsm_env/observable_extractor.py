from ...poe_world.core import (
    ObservableExtractorProtocol,
    ObservableId,
    DiscreteDistribution,
)
from ...deterministic_fsm_env import State, Action
from ...typing_utils import implements
import numpy as np
from ..optimization import combine_expert_predictions_for_attr
import torch


class ObservableExtractor:
    def __init__(self):
        self.switch_a_domain = np.array([0, 1])
        self.switch_b_domain = np.array([0, 1])

    def extract_attribute_predictions(
        self, state: State
    ) -> dict[ObservableId, DiscreteDistribution]:
        predictions: dict[ObservableId, DiscreteDistribution] = {}

        if isinstance(state.switch_a, DiscreteDistribution):
            predictions[ObservableId("switch_a")] = state.switch_a.expand_support(
                self.switch_a_domain
            )

        if isinstance(state.switch_b, DiscreteDistribution):
            predictions[ObservableId("switch_b")] = state.switch_b.expand_support(
                self.switch_b_domain
            )

        return predictions

    def get_observed_outcomes(self, state: State) -> dict[ObservableId, int]:
        return {
            ObservableId("switch_a"): state.switch_a,
            ObservableId("switch_b"): state.switch_b,
        }

    def apply_expert_predictions(
        self,
        new_state: State,
        expert_predictions: dict[ObservableId, list[DiscreteDistribution]],
        weights: torch.Tensor,
    ) -> State:

        if "switch_a" in expert_predictions:
            switch_a_preds = expert_predictions[ObservableId("switch_a")]
            combined_dist = combine_expert_predictions_for_attr(switch_a_preds, weights)
            new_state.switch_a = combined_dist.sample()

        if "switch_b" in expert_predictions:
            switch_b_preds = expert_predictions[ObservableId("switch_b")]
            combined_dist = combine_expert_predictions_for_attr(switch_b_preds, weights)
            new_state.switch_b = combined_dist.sample()

        return new_state


implements(ObservableExtractorProtocol[State])(ObservableExtractor)
