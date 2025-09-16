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


class CorrectCombatDamageLaw:
    def __init__(self, focus: str = "all"):
        self.focus = focus

    def precondition(self, current_state: WorldState, action: CrafterAction) -> bool:
        return action == "do"

    def effect(self, current_state: WorldState, action: CrafterAction) -> None:
        facing_x = current_state.player.position.x + current_state.player.facing.x
        facing_y = current_state.player.position.y + current_state.player.facing.y

        # Check if there's an entity at the facing position
        target_entity = None
        for entity in current_state.objects:
            if (
                entity.position.x == facing_x
                and entity.position.y == facing_y
                and entity.entity_id != current_state.player.entity_id
            ):
                target_entity = entity
                break

        if target_entity:
            # Check if we should focus on this entity type
            if self.focus != "all" and target_entity.name != self.focus:
                return  # Skip this entity if it doesn't match the focus

            # Calculate damage based on player's weapons
            base_damage = 1
            if current_state.player.inventory.wood_sword > 0:
                base_damage = 2
            elif current_state.player.inventory.stone_sword > 0:
                base_damage = 3
            elif current_state.player.inventory.iron_sword > 0:
                base_damage = 5

            # Apply damage to target entity
            new_health = max(0, target_entity.health - base_damage)
            target_entity.health = DiscreteDistribution(support=[new_health])  # type: ignore


class IncorrectCombatDamageExpertInstakills:
    def precondition(self, current_state: WorldState, action: CrafterAction) -> bool:
        return action == "do"

    def effect(self, current_state: WorldState, action: CrafterAction) -> None:
        """
        Incorrect expert that instantly kills any entity when attacked.

        This expert is obviously wrong - any 'do' action against an entity
        immediately sets their health to 0, regardless of weapons or damage.

        Args:
            current_state: The current game state to modify in-place
            action: The action being taken
            **context: Additional context (unused)
        """

        # Get the tile the player is facing
        facing_x = current_state.player.position.x + current_state.player.facing.x
        facing_y = current_state.player.position.y + current_state.player.facing.y

        # Check if there's an entity at the facing position
        target_entity = None
        for entity in current_state.objects:
            if (
                entity.position.x == facing_x
                and entity.position.y == facing_y
                and entity.entity_id != current_state.player.entity_id
            ):
                target_entity = entity
                break

        if target_entity:
            # Instantly kill the entity (obviously wrong!)
            target_entity.health = DiscreteDistribution(support=[0])  # type: ignore


CORRECT_EXPERTS = [
    LawFunctionWrapper.from_non_runtime_created(CorrectPlayerMovementLaw()),
    LawFunctionWrapper.from_non_runtime_created(CorrectCombatDamageLaw()),
]

INCORRECT_EXPERTS = [
    LawFunctionWrapper.from_non_runtime_created(
        IncorrectPlayerMovementExpertTeleports()
    ),
    LawFunctionWrapper.from_non_runtime_created(
        IncorrectCombatDamageExpertInstakills()
    ),
]

ALL_EXPERTS = CORRECT_EXPERTS + INCORRECT_EXPERTS
