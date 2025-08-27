"""
PoE-World was originally designed for 2D physics-based environments that are
object-centric, like Atari Pong or 2D platformers like Montezuma's Revenge.

The "state" in these environments is a list of objects with commmon attributes
like position, velocity, etc.

In Crafter, the state is much more complex, and hierarchical. Since our aim is
to evaluate the ability of a more generic approach to learn about the world, we
want to avoid hardcoding domain knowledge about the world, such as how to extract
interesting observables from the state and so on.

So, we will stick to the original design of PoE-World, and define an extractor which
operates on objects in the world state that are similar to the objects in the original
environments.

This corresponds to the player's position, health, and the position and health of nearby
game entities. Like in PoE-World for physics-based environments, we will ignore static
objects like tiles of the game world or crafting stations.
"""

from ...typing_utils import implements
from crafter.state_export import WorldState
from ..core import ObservableExtractorProtocol, ObservableId, DiscreteDistribution
import torch
import numpy as np


class ObservableExtractor:
    def __init__(self):
        # Define fixed domains for position and health attributes
        # Using reasonable defaults without reading from game state
        self.position_domain = np.arange(0, 101)  # [0, 1, 2, ..., 100]
        self.health_domain = np.arange(0, 101)  # [0, 1, 2, ..., 100]

    def extract_attribute_predictions(
        self, state: WorldState
    ) -> dict[ObservableId, DiscreteDistribution]:
        """
        Extract probabilistic predictions from a state after expert execution.
        """
        predictions: dict[ObservableId, DiscreteDistribution] = {}

        # Extract player position
        if hasattr(state.player.position, "x") and isinstance(
            state.player.position.x, DiscreteDistribution
        ):
            predictions[ObservableId("player_position_x")] = (
                state.player.position.x.expand_support(self.position_domain)
            )
        else:
            predictions[ObservableId("player_position_x")] = (
                DiscreteDistribution.from_uniform(self.position_domain)
            )

        if hasattr(state.player.position, "y") and isinstance(
            state.player.position.y, DiscreteDistribution
        ):
            predictions[ObservableId("player_position_y")] = (
                state.player.position.y.expand_support(self.position_domain)
            )
        else:
            predictions[ObservableId("player_position_y")] = (
                DiscreteDistribution.from_uniform(self.position_domain)
            )

        # Extract player health
        if isinstance(state.player.health, DiscreteDistribution):
            predictions[ObservableId("player_health")] = (
                state.player.health.expand_support(self.health_domain)
            )
        else:
            predictions[ObservableId("player_health")] = (
                DiscreteDistribution.from_uniform(self.health_domain)
            )

        # Extract entity positions and health
        for entity in state.objects:
            if entity.entity_id == state.player.entity_id:
                continue  # Skip player as we already handled it

            # Entity position x
            attr_name = f"entity_{entity.entity_id}_position_x"
            if hasattr(entity.position, "x") and isinstance(
                entity.position.x, DiscreteDistribution
            ):
                predictions[ObservableId(attr_name)] = entity.position.x.expand_support(
                    self.position_domain
                )
            else:
                predictions[ObservableId(attr_name)] = (
                    DiscreteDistribution.from_uniform(self.position_domain)
                )

            # Entity position y
            attr_name = f"entity_{entity.entity_id}_position_y"
            if hasattr(entity.position, "y") and isinstance(
                entity.position.y, DiscreteDistribution
            ):
                predictions[ObservableId(attr_name)] = entity.position.y.expand_support(
                    self.position_domain
                )
            else:
                predictions[ObservableId(attr_name)] = (
                    DiscreteDistribution.from_uniform(self.position_domain)
                )

            # Entity health
            attr_name = f"entity_{entity.entity_id}_health"
            if isinstance(entity.health, DiscreteDistribution):
                predictions[ObservableId(attr_name)] = entity.health.expand_support(
                    self.health_domain
                )
            else:
                predictions[ObservableId(attr_name)] = (
                    DiscreteDistribution.from_uniform(self.health_domain)
                )

        return predictions

    def get_observed_outcomes(self, state: WorldState) -> dict[ObservableId, int]:
        """
        Extract ground truth observed values from a state.
        """
        observed: dict[ObservableId, int] = {}

        # Player position
        observed[ObservableId("player_position_x")] = state.player.position.x
        observed[ObservableId("player_position_y")] = state.player.position.y

        # Player health
        observed[ObservableId("player_health")] = state.player.health

        # Entity positions and health
        for entity in state.objects:
            if entity.entity_id == state.player.entity_id:
                continue  # Skip player as we already handled it

            observed[ObservableId(f"entity_{entity.entity_id}_position_x")] = (
                entity.position.x
            )
            observed[ObservableId(f"entity_{entity.entity_id}_position_y")] = (
                entity.position.y
            )
            observed[ObservableId(f"entity_{entity.entity_id}_health")] = entity.health

        return observed

    @staticmethod
    def apply_expert_predictions(
        new_state: WorldState,
        expert_predictions: dict[ObservableId, list[DiscreteDistribution]],
        weights: torch.Tensor,
    ) -> WorldState:
        """
        Apply combined expert predictions to create a new state.
        """
        from ..weight_fitter import combine_expert_predictions_for_attr

        # Sample player position
        if "player_position_x" in expert_predictions:
            player_x_preds = expert_predictions[ObservableId("player_position_x")]
            combined_dist = combine_expert_predictions_for_attr(player_x_preds, weights)
            new_state.player.position.x = combined_dist.sample()

        if "player_position_y" in expert_predictions:
            player_y_preds = expert_predictions[ObservableId("player_position_y")]
            combined_dist = combine_expert_predictions_for_attr(player_y_preds, weights)
            new_state.player.position.y = combined_dist.sample()

        # Sample player health
        if "player_health" in expert_predictions:
            player_health_preds = expert_predictions[ObservableId("player_health")]
            combined_dist = combine_expert_predictions_for_attr(
                player_health_preds, weights
            )
            new_state.player.health = combined_dist.sample()

        # Sample entity positions and health
        for entity in new_state.objects:
            if entity.entity_id == new_state.player.entity_id:
                continue  # Skip player as we already handled it

            # Entity position x
            attr_name = f"entity_{entity.entity_id}_position_x"
            if attr_name in expert_predictions:
                entity_x_preds = expert_predictions[ObservableId(attr_name)]
                combined_dist = combine_expert_predictions_for_attr(
                    entity_x_preds, weights
                )
                entity.position.x = combined_dist.sample()

            # Entity position y
            attr_name = f"entity_{entity.entity_id}_position_y"
            if attr_name in expert_predictions:
                entity_y_preds = expert_predictions[ObservableId(attr_name)]
                combined_dist = combine_expert_predictions_for_attr(
                    entity_y_preds, weights
                )
                entity.position.y = combined_dist.sample()

            # Entity health
            attr_name = f"entity_{entity.entity_id}_health"
            if attr_name in expert_predictions:
                entity_health_preds = expert_predictions[ObservableId(attr_name)]
                combined_dist = combine_expert_predictions_for_attr(
                    entity_health_preds, weights
                )
                entity.health = combined_dist.sample()

        return new_state


implements(ObservableExtractorProtocol[WorldState])(ObservableExtractor)
