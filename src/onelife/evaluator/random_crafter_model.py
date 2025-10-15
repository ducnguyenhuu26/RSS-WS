"""
Random baseline world model for Crafter game states.

This module provides a random world model that generates random predictions
for next states using bounds and constraints defined in the game constants.
"""

import random
import copy
from typing import Optional, cast, Union

from .core import EvaluatableWorldModel
from ..typing_utils import implements
from crafter.state_export import (
    WorldState,
    PlayerState,
    CowState,
    ZombieState,
    SkeletonState,
    ArrowState,
    PlantState,
    FenceState,
    Position,
    Inventory,
    Achievements,
    BaseObjectState,
)
from crafter.constants import (
    materials,
    items,
    ActionT,
)


class RandomCrafterWorldModel:
    """Random baseline world model for Crafter.

    This model generates random predictions for the next world state,
    using bounds and constraints from the game constants.
    """

    def __init__(self, rng: Optional[random.Random] = None):
        """Initialize the random world model.

        Args:
            rng: Random number generator to use. If None, uses default random.
        """
        self.rng = rng or random.Random()

    def sample_next_state(
        self, current_state: WorldState, action: ActionT
    ) -> WorldState:
        """Generate a random prediction for the next world state.

        This method creates a randomized version of the world state by:
        - Keeping structural properties (size, chunk_size, view) unchanged
        - Randomizing materials grid with possible material types
        - Randomizing object properties within reasonable bounds
        - Randomizing player state properties
        - Randomizing environmental properties like daylight

        Args:
            current_state: The current world state
            action: The action taken (ignored for random model)

        Returns:
            A randomly generated next world state
        """
        # Create a deep copy to avoid modifying the input
        next_state = copy.deepcopy(current_state)

        # Randomize materials grid
        next_state.materials = self._randomize_materials(
            next_state.size[0], next_state.size[1]
        )

        # Randomize objects
        next_state.objects = cast(
            list[
                Union[
                    PlayerState,
                    CowState,
                    ZombieState,
                    SkeletonState,
                    ArrowState,
                    PlantState,
                    FenceState,
                ]
            ],
            [
                self._randomize_object(obj, next_state.size)
                for obj in next_state.objects
            ],
        )

        # Randomize player state
        next_state.player = self._randomize_player_state(
            next_state.player, next_state.size
        )

        # Randomize environmental properties
        next_state.daylight = self.rng.random()  # 0.0 to 1.0
        next_state.step_count = max(0, next_state.step_count + self.rng.randint(-5, 10))

        # Randomize event bus (could be empty or have random events)
        next_state.event_bus = (
            [f"random_event_{self.rng.randint(0, 100)}"]
            if self.rng.random() < 0.1
            else []
        )

        return next_state

    def evaluate_log_probability(
        self, state: WorldState, action: ActionT, next_state: WorldState
    ) -> float:
        """Compute log probability of next_state given state and action.

        For the random model, all transitions have equal probability,
        so we return 0.0 (equivalent to probability 1 in log space for
        multiple choice evaluation).

        Args:
            state: Current world state
            action: Action taken
            next_state: Predicted next world state

        Returns:
            Log probability (always 0.0 for random model)
        """
        return 0.0

    def _randomize_materials(
        self, width: int, height: int
    ) -> list[list[Optional[str]]]:
        """Generate a random materials grid.

        Args:
            width: Grid width
            height: Grid height

        Returns:
            2D list of random materials
        """
        grid = []
        for x in range(width):
            row = []
            for y in range(height):
                # Randomly choose a material or None
                if self.rng.random() < 0.8:  # 80% chance of having a material
                    row.append(self.rng.choice(materials))
                else:
                    row.append(None)
            grid.append(row)
        return grid

    def _randomize_object(
        self, obj: BaseObjectState, world_size: tuple[int, int]
    ) -> BaseObjectState:
        """Randomize an object's properties.

        Args:
            obj: The object to randomize
            world_size: Size of the world for position bounds

        Returns:
            Object with randomized properties
        """
        # Randomize position within world bounds
        obj.position = Position(
            x=self.rng.randint(0, world_size[0] - 1),
            y=self.rng.randint(0, world_size[1] - 1),
        )

        # Randomize health (0-9 for most objects)
        obj.health = self.rng.randint(0, 9)

        # Randomize removed status (10% chance)
        obj.removed = self.rng.random() < 0.1

        # Object-specific randomization
        if isinstance(obj, PlayerState):
            return self._randomize_player_state(obj, world_size)
        elif isinstance(obj, ZombieState):
            obj.cooldown = self.rng.randint(0, 10)  # Arbitrary cooldown range
        elif isinstance(obj, SkeletonState):
            obj.reload = self.rng.randint(0, 5)  # Arbitrary reload range
        elif isinstance(obj, ArrowState):
            # Randomize facing direction
            obj.facing = Position(
                x=self.rng.choice([-1, 0, 1]),
                y=self.rng.choice([-1, 0, 1]),
            )
        elif isinstance(obj, PlantState):
            obj.grown = self.rng.randint(0, 10)  # Arbitrary growth stages
            obj.ripe = self.rng.random() < 0.5
        # CowState and FenceState have no additional fields to randomize

        return obj

    def _randomize_player_state(
        self, player: PlayerState, world_size: tuple[int, int]
    ) -> PlayerState:
        """Randomize player-specific properties.

        Args:
            player: Player state to randomize
            world_size: World size for position bounds

        Returns:
            Player with randomized properties
        """
        # Randomize position
        player.position = Position(
            x=self.rng.randint(0, world_size[0] - 1),
            y=self.rng.randint(0, world_size[1] - 1),
        )

        # Randomize facing direction
        player.facing = Position(
            x=self.rng.choice([-1, 0, 1]),
            y=self.rng.choice([-1, 0, 1]),
        )

        # Randomize health and inventory
        player.health = self.rng.randint(0, items["health"]["max"])
        player.inventory = self._randomize_inventory()

        # Randomize achievements
        player.achievements = self._randomize_achievements()

        # Randomize physiological stats (0.0 to 1.0)
        player.thirst = self.rng.random()
        player.hunger = self.rng.random()
        player.fatigue = self.rng.random()
        player.recover = self.rng.random()

        # Randomize sleeping status (rare)
        player.sleeping = self.rng.random() < 0.05

        # Randomize last_health
        player.last_health = self.rng.randint(0, 9)

        return player

    def _randomize_inventory(self) -> Inventory:
        """Generate a random inventory within item limits.

        Returns:
            Random inventory
        """
        return Inventory(
            health=self.rng.randint(0, items["health"]["max"]),
            food=self.rng.randint(0, items["food"]["max"]),
            drink=self.rng.randint(0, items["drink"]["max"]),
            energy=self.rng.randint(0, items["energy"]["max"]),
            sapling=self.rng.randint(0, items["sapling"]["max"]),
            wood=self.rng.randint(0, items["wood"]["max"]),
            stone=self.rng.randint(0, items["stone"]["max"]),
            coal=self.rng.randint(0, items["coal"]["max"]),
            iron=self.rng.randint(0, items["iron"]["max"]),
            diamond=self.rng.randint(0, items["diamond"]["max"]),
            wood_pickaxe=self.rng.randint(0, items["wood_pickaxe"]["max"]),
            stone_pickaxe=self.rng.randint(0, items["stone_pickaxe"]["max"]),
            iron_pickaxe=self.rng.randint(0, items["iron_pickaxe"]["max"]),
            wood_sword=self.rng.randint(0, items["wood_sword"]["max"]),
            stone_sword=self.rng.randint(0, items["stone_sword"]["max"]),
            iron_sword=self.rng.randint(0, items["iron_sword"]["max"]),
        )

    def _randomize_achievements(self) -> Achievements:
        """Generate random achievement progress.

        Returns:
            Random achievements
        """
        return Achievements(
            collect_coal=self.rng.randint(0, 10),
            collect_diamond=self.rng.randint(0, 10),
            collect_drink=self.rng.randint(0, 10),
            collect_iron=self.rng.randint(0, 10),
            collect_sapling=self.rng.randint(0, 10),
            collect_stone=self.rng.randint(0, 10),
            collect_wood=self.rng.randint(0, 10),
            defeat_skeleton=self.rng.randint(0, 10),
            defeat_zombie=self.rng.randint(0, 10),
            eat_cow=self.rng.randint(0, 10),
            eat_plant=self.rng.randint(0, 10),
            make_iron_pickaxe=self.rng.randint(0, 1),  # Binary achievements
            make_iron_sword=self.rng.randint(0, 1),
            make_stone_pickaxe=self.rng.randint(0, 1),
            make_stone_sword=self.rng.randint(0, 1),
            make_wood_pickaxe=self.rng.randint(0, 1),
            make_wood_sword=self.rng.randint(0, 1),
            place_furnace=self.rng.randint(0, 1),
            place_plant=self.rng.randint(0, 10),
            place_stone=self.rng.randint(0, 10),
            place_table=self.rng.randint(0, 10),
            wake_up=self.rng.randint(0, 10),
        )


# Register the implementation
implements(EvaluatableWorldModel[WorldState, ActionT])(RandomCrafterWorldModel)
