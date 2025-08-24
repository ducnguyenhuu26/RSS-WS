"""
Scenario definitions for Crafter evaluation.
"""

from typing import Protocol
from crafter.functional_env import (
    reconstruct_world_from_state,
    export_world_state,
)
from crafter.state_export import WorldState
from crafter.constants import ActionT
from .utils import find_player
from crafter.testing_helpers import (
    player_utils,
    world_utils,
)
from crafter.functional_env import initial_state
from crafter import objects
from ...typing_utils import implements


class Scenario(Protocol):
    """Protocol for scenario definitions."""

    @property
    def name(self) -> str:
        """The name of this scenario."""
        ...

    def get_initial_state(self) -> WorldState:
        """Creates and returns the specific starting WorldState for this scenario."""
        ...

    def get_actions(self) -> list[ActionT]:
        """Returns the sequence of actions to execute for this scenario."""
        ...


class CraftWoodenPickaxeScenario:
    """Scenario for testing crafting a wooden pickaxe."""

    @property
    def name(self) -> str:
        return "craft_wooden_pickaxe"

    def get_initial_state(self) -> WorldState:
        """
        Creates a temporary environment, configures it to the desired
        starting conditions, and returns the resulting WorldState.
        """
        view = (9, 9)
        state = initial_state(area=(9, 9), view=view, seed=1)
        world = reconstruct_world_from_state(state)

        player = find_player(world)
        player_utils.set_player_position(player, (5, 5))
        player_utils.set_player_facing(player, (0, 1))
        world_utils.set_tile_material(world, (5, 6), "table")
        player_utils.set_player_inventory_item(player, "wood", 2)
        player_utils.set_player_inventory_item(player, "wood_pickaxe", 0)

        return export_world_state(world, view=view, step_count=0)

    def get_actions(self) -> list[ActionT]:
        return ["make_wood_pickaxe"]


implements(Scenario)(CraftWoodenPickaxeScenario)


class CowMovementScenario:
    """Scenario for testing cow movement behavior."""

    @property
    def name(self) -> str:
        return "cow_movement"

    def get_initial_state(self) -> WorldState:
        """
        Creates a temporary environment with a cow near the player.
        """
        view = (9, 9)
        state = initial_state(area=(9, 9), view=view, seed=1)
        world = reconstruct_world_from_state(state)

        # Clear all the other tiles around the world to be grass
        for x in range(view[0]):
            for y in range(view[1]):
                world_utils.set_tile_material(world, (x, y), "grass")

        player = find_player(world)
        player_utils.set_player_position(player, (5, 5))

        # Clear all entities from the world (except the player)
        for obj in world.objects:
            if isinstance(obj, objects.Player):
                continue
            world.remove(obj)

        # Add a cow near the player
        cow = objects.Cow(world, (6, 6))
        world.add(cow)

        return export_world_state(world, view=view, step_count=0)

    def get_actions(self) -> list[ActionT]:
        return ["noop"] * 1


implements(Scenario)(CowMovementScenario)
