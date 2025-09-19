import pytest

from crafter.functional_env import initial_state
from crafter.state_reconstruction import reconstruct_world_from_state
from crafter.state_export import export_world_state
from crafter import objects as crafter_objects
from crafter.testing_helpers.world_utils import (
    set_tile_material,
    add_object_to_world,
    remove_object_from_world,
)
from crafter.testing_helpers.player_utils import (
    set_player_facing,
)
from distant_sunburn.unsupervised_crafter_env_factory import TextRendererOutput


@pytest.fixture
def base_state():
    state = initial_state(area=(32, 32), view=(9, 9), episode=1, seed=123)
    return state


def _place_material(world, x, y, material):
    world[(x, y)] = material


def _add_object(world, obj):
    world.add(obj)


def _center_pos(world):
    return world.area[0] // 2, world.area[1] // 2


class DummyRenderer:
    """
    Placeholder import guard for type compatibility; real renderer is implemented
    in distant_sunburn.unsupervised_crafter_env_factory. Tests will import the real one.
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, state):
        raise NotImplementedError


@pytest.mark.parametrize("k_nearest_outside", [1, 2])
def test_local_view_and_nearest_outside_summary(base_state, k_nearest_outside):
    # Import here to avoid import cycles during collection
    from distant_sunburn.unsupervised_crafter_env_factory import (
        UnsupervisedTextRenderer,
    )

    # Reconstruct world and manipulate
    world = reconstruct_world_from_state(base_state)

    # Player at center
    px, py = _center_pos(world)

    # Inside local view (update_range computed from state.view)
    set_tile_material(world, (px + 2, py), "tree")
    cow = add_object_to_world(world, crafter_objects.Cow, (px - 3, py + 1))

    # Outside local view but inside bounds: add a furnace and a diamond
    update_range = base_state.update_range
    # pick a delta that is just outside local view but clamped to world size
    size_x, size_y = world.area
    fx = min(size_x - 1, px + update_range + 1)
    fy = py
    if fx == px and fy == py:
        fx = min(size_x - 1, px + update_range + 2)
    dx = max(0, px - (update_range + 1))
    dy = max(0, py - (update_range + 1))
    set_tile_material(world, (fx, fy), "furnace")
    set_tile_material(world, (dx, dy), "diamond")

    # Re-export state
    state = export_world_state(
        world, view=base_state.view, step_count=base_state.step_count
    )

    renderer = UnsupervisedTextRenderer(
        report_k_nearest_outside=k_nearest_outside,
        include_zero_inventory=True,
    )
    out: TextRendererOutput = renderer(state)

    # Short-term context contains local view items with relative and absolute coords
    st = out.short_term_context
    assert "tree" in st and "(x=" in st and "y=" in st
    assert "cow" in st and (
        "west" in st or "east" in st or "north" in st or "south" in st
    )

    # Long-term context contains outside summary including furnace and diamond
    lt = out.long_term_context
    assert "outside view" in lt
    assert "furnace" in lt
    assert "diamond" in lt

    # If a crafting station is not present (e.g., table), we mention not present
    if "table" not in lt:
        assert "table: not present" in lt


def test_inventory_and_status_and_facing_targeting(base_state):
    from distant_sunburn.unsupervised_crafter_env_factory import (
        UnsupervisedTextRenderer,
    )

    world = reconstruct_world_from_state(base_state)
    px, py = _center_pos(world)

    # Face east toward a zombie on grass
    player = None
    for obj in world.objects:
        if isinstance(obj, crafter_objects.Player):
            player = obj
            break
    assert player is not None

    set_player_facing(player, (1, 0))  # east
    # Ensure grass in front and a zombie there
    front = (px + 1, py)
    set_tile_material(world, front, "grass")
    zombie = add_object_to_world(world, crafter_objects.Zombie, front, player)

    # Make inventory explicit: include zeros
    # The renderer should show zero counts
    player.inventory["wood"] = 0
    player.inventory["stone"] = 0

    state = export_world_state(
        world, view=base_state.view, step_count=base_state.step_count
    )

    renderer = UnsupervisedTextRenderer(include_zero_inventory=True)
    out: TextRendererOutput = renderer(state)

    # Status is shown (sleeping/dead not true, but status header should exist)
    assert "Status" in out.long_term_context

    # Facing direction and targeting phrase
    assert "facing: east" in out.long_term_context
    assert "you are targeting a zombie on grass" in out.long_term_context

    # Inventory includes zeros for visibility
    assert "wood: 0" in out.short_term_context
    assert "stone: 0" in out.short_term_context

    # If we change target to empty grass, targeting a patch of grass
    world = reconstruct_world_from_state(state)
    # Remove the zombie and leave grass
    for obj in list(world.objects):
        if isinstance(obj, crafter_objects.Zombie):
            remove_object_from_world(world, obj)
    state2 = export_world_state(
        world, view=base_state.view, step_count=base_state.step_count
    )
    out2: TextRendererOutput = renderer(state2)
    assert "you are targeting a patch of grass" in out2.long_term_context
