"""
Utility functions for Crafter evaluation.
"""

from crafter.env import Env
from crafter.state_export import WorldState, export_world_state
from crafter.engine import World
import crafter.objects as crafter_objects


def get_world_state(env: Env) -> WorldState:
    """Exports the WorldState from a crafter.Env instance."""
    assert env._step is not None
    return export_world_state(env._world, view=env._config.view, step_count=env._step)


def find_player(world: World) -> crafter_objects.Player:
    """Finds the player in the world."""
    for obj in world.objects:
        if isinstance(obj, crafter_objects.Player):
            return obj
    raise ValueError("No player found in world")
