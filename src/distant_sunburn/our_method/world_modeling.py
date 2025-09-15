import copy
from typing import Dict, Generic, TypeVar

import torch

from distant_sunburn.poe_world.core import ObservableExtractorProtocol

from ..typing_utils import implements
from ..poe_world.core import (
    DiscreteDistribution,
    ObservableId,
)
from .optimization import (
    combine_expert_predictions_for_attr,
)
from .core import WeightedLaw, WorldModelProtocol

# Constants for log probability values
LOG_IMPOSSIBLE_VALUE = -1000.0  # Very low probability for impossible transitions

SymbolicStateT = TypeVar("SymbolicStateT")
ActionT = TypeVar("ActionT")


class LawMixture(Generic[SymbolicStateT, ActionT]):
    def __init__(
        self,
        observable_extractor: ObservableExtractorProtocol[SymbolicStateT],
        weighted_laws: list[WeightedLaw[SymbolicStateT]] | None = None,
    ):
        self._laws = weighted_laws or []
        self.observable_extractor = observable_extractor

    @property
    def laws(self) -> list[WeightedLaw[SymbolicStateT]]:
        """Get the list of weighted laws."""
        return self._laws

    def with_new_laws(
        self, new_laws: list[WeightedLaw[SymbolicStateT]]
    ) -> "LawMixture[SymbolicStateT, ActionT]":
        """Create a new world model with the given laws."""
        return LawMixture(
            weighted_laws=new_laws, observable_extractor=self.observable_extractor
        )

    def sample_next_state(
        self, current_state: SymbolicStateT, action: ActionT
    ) -> SymbolicStateT:
        """
        Sample a next state using the weighted laws.

        Args:
            current_state: Current game state
            action: Action being taken

        Returns:
            Sampled next state
        """
        if not self._laws:
            # No experts - return current state unchanged
            return copy.deepcopy(current_state)

        # Get expert predictions
        expert_predictions = self._get_law_predictions(current_state, action)

        # Extract weights as tensor
        weights = torch.tensor(
            [expert.weight for expert in self._laws], dtype=torch.float32
        )

        # Create new state by sampling from combined distributions
        new_state = copy.deepcopy(current_state)

        new_state = self.observable_extractor.apply_expert_predictions(
            new_state, expert_predictions, weights
        )

        return new_state

    def evaluate_log_probability(
        self, state: SymbolicStateT, action: ActionT, next_state: SymbolicStateT
    ) -> float:
        """
        Evaluate the log-probability of a transition under this model.

        Args:
            state: Current state
            action: Action taken
            next_state: Next state

        Returns:
            Log-probability of the transition
        """
        if not self._laws:
            return LOG_IMPOSSIBLE_VALUE

        # Get expert predictions
        expert_predictions = self._get_law_predictions(state, action)

        # Extract weights as tensor
        weights = torch.tensor(
            [expert.weight for expert in self._laws], dtype=torch.float32
        )

        # Get observed values from next state
        observed_values = self.observable_extractor.get_observed_outcomes(next_state)

        total_log_prob = 0.0

        # Evaluate log-probability for each attribute
        for attr_name, observed_value in observed_values.items():
            if attr_name in expert_predictions:
                attr_predictions = expert_predictions[attr_name]
                combined_dist = combine_expert_predictions_for_attr(
                    attr_predictions, weights
                )
                log_prob = combined_dist.evaluate_log_probability(observed_value)
                total_log_prob += log_prob

        return total_log_prob

    def _get_law_predictions(
        self, state: SymbolicStateT, action: ActionT
    ) -> Dict[ObservableId, list[DiscreteDistribution]]:
        """
        Get predictions from all experts for the given state and action.

        Returns:
            Dictionary mapping attribute names to lists of law predictions
        """
        predictions_from_all_laws: dict[ObservableId, list[DiscreteDistribution]] = {}

        for weighted_law in self._laws:
            # Deep copy state and run expert
            state_copy = copy.deepcopy(state)
            weighted_law.law.effect(state_copy, action)

            # Extract predictions for each attribute
            attr_predictions = self.observable_extractor.extract_attribute_predictions(
                state_copy
            )

            # Group by attribute name
            for attr_name, attr_prediction in attr_predictions.items():
                if attr_name not in predictions_from_all_laws:
                    predictions_from_all_laws[attr_name] = []
                predictions_from_all_laws[attr_name].append(attr_prediction)

        return predictions_from_all_laws


implements(WorldModelProtocol)(LawMixture)
