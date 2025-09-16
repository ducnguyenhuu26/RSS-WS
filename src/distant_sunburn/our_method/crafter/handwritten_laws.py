from ..core import LawFunctionWrapper
from crafter.state_export import WorldState
from crafter.constants import ActionT as CrafterAction
from loguru import logger
from ...poe_world.core import DiscreteDistribution


class CorrectPlayerMovementLaw:
    def precondition(self, current_state: WorldState, action: CrafterAction) -> bool:
        return action in {"move_left", "move_right", "move_up", "move_down"}

    def effect(self, current_state: WorldState, action: CrafterAction) -> None:
        """
        Correct expert that models player movement mechanics.

        This expert models:
        - Player moves in the direction of the action (left, right, up, down)
        - Movement is bounded by world boundaries
        - Player position is updated immediately for movement actions
        - Non-movement actions don't change player position

        Args:
            current_state: The current game state to modify in-place
            action: The action being taken
            **context: Additional context (unused)
        """
        # Only movement actions change player position

        # Calculate new position based on action
        new_x = current_state.player.position.x
        new_y = current_state.player.position.y

        if action == "move_left":
            new_x = max(0, new_x - 1)
        elif action == "move_right":
            new_x = min(current_state.size[0] - 1, new_x + 1)
        elif action == "move_up":
            new_y = max(0, new_y - 1)
        elif action == "move_down":
            new_y = min(current_state.size[1] - 1, new_y + 1)

        # Assign predictions using DiscreteDistribution
        current_state.player.position.x = DiscreteDistribution(support=[new_x])  # type: ignore
        current_state.player.position.y = DiscreteDistribution(support=[new_y])  # type: ignore


class IncorrectPlayerMovementExpertTeleports:
    def precondition(self, current_state: WorldState, action: CrafterAction) -> bool:
        return action in {"move_left", "move_right", "move_up", "move_down"}

    def effect(self, current_state: WorldState, action: CrafterAction) -> None:
        rng = current_state.random_state
        new_x = rng.randint(0, current_state.size[0] - 1)
        new_y = rng.randint(0, current_state.size[1] - 1)

        current_state.player.position.x = DiscreteDistribution(support=[new_x])  # type: ignore
        current_state.player.position.y = DiscreteDistribution(support=[new_y])  # type: ignore


CORRECT_EXPERTS = [
    LawFunctionWrapper.from_non_runtime_created(CorrectPlayerMovementLaw()),
]

INCORRECT_EXPERTS = [
    LawFunctionWrapper.from_non_runtime_created(
        IncorrectPlayerMovementExpertTeleports()
    ),
]

ALL_EXPERTS = CORRECT_EXPERTS + INCORRECT_EXPERTS
