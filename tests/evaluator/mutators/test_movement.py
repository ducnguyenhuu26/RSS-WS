from distant_sunburn.evaluator.crafter.mutators.movement import (
    IllegalMovementMutator,
    NON_MOVEMENT_ACTIONS,
    MOVEMENT_ACTIONS,
)
from crafter.state_export import WorldState
from crafter.functional_env import (
    initial_state,
    reconstruct_world_from_state,
    export_world_state,
    transition,
)
from crafter.testing_helpers import player_utils, world_utils
from distant_sunburn.evaluator.crafter.utils import find_player
from distant_sunburn.evaluator.crafter.utils import MAP_ACTION_TO_INDEX
import copy
import pytest
from crafter.constants import ActionT
from crafter import engine as crafter_engine


@pytest.fixture
def grass_field_state() -> tuple[WorldState, crafter_engine.World]:
    view = (9, 9)
    state = initial_state(area=(9, 9), view=view, seed=1)
    world = reconstruct_world_from_state(state)

    player = find_player(world)
    player_utils.set_player_position(player, (5, 5))

    # Clear all the other tiles around the world to be grass
    for x in range(view[0]):
        for y in range(view[1]):
            world_utils.set_tile_material(world, (x, y), "grass")

    return export_world_state(world, view=view, step_count=0), world


@pytest.mark.parametrize("action", NON_MOVEMENT_ACTIONS)
def test_non_movement_actions(
    grass_field_state: tuple[WorldState, crafter_engine.World], action: ActionT
):
    state = grass_field_state[0]

    true_next_state, _ = transition(copy.deepcopy(state), MAP_ACTION_TO_INDEX[action])

    mutator = IllegalMovementMutator()

    assert mutator.precondition(state, action)

    mutated_next_state = mutator(state, action)

    assert mutated_next_state.player.position != true_next_state.player.position


@pytest.mark.parametrize("action", MOVEMENT_ACTIONS)
def test_movement_actions(
    grass_field_state: tuple[WorldState, crafter_engine.World], action: ActionT
):
    state = grass_field_state[0]

    true_next_state, _ = transition(copy.deepcopy(state), MAP_ACTION_TO_INDEX[action])

    mutator = IllegalMovementMutator()

    assert not mutator.precondition(state, action)
