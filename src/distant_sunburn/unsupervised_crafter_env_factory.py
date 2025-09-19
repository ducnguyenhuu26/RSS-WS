from dataclasses import dataclass

import crafter.constants
import numpy as np
from crafter.env import Env as BaseCrafterEnv
from crafter.state_export import WorldState, export_world_state
from PIL import Image
from pydantic import ConfigDict

from distant_sunburn.balrog_components import EnvironmentConfig
from pydantic import Field

from .balrog_interfaces import (
    EnvironmentProtocol,
    Experience,
    Observation,
    OnResetExperience,
    Text,
)
from .typing_utils import implements
from typing import Optional
from typing import Protocol, Callable
from crafter.state_export import Position

MAP_SLUG_TO_ENGINE_ACTION = {
    slug: idx for idx, slug in enumerate(crafter.constants.actions)
}


MAP_DISPLAY_ACTION_TO_ENGINE_ACTION = {
    "Noop": MAP_SLUG_TO_ENGINE_ACTION["noop"],
    "Move West": MAP_SLUG_TO_ENGINE_ACTION["move_left"],
    "Move East": MAP_SLUG_TO_ENGINE_ACTION["move_right"],
    "Move North": MAP_SLUG_TO_ENGINE_ACTION["move_up"],
    "Move South": MAP_SLUG_TO_ENGINE_ACTION["move_down"],
    "Do": MAP_SLUG_TO_ENGINE_ACTION["do"],
    "Sleep": MAP_SLUG_TO_ENGINE_ACTION["sleep"],
    "Place Stone": MAP_SLUG_TO_ENGINE_ACTION["place_stone"],
    "Place Table": MAP_SLUG_TO_ENGINE_ACTION["place_table"],
    "Place Furnace": MAP_SLUG_TO_ENGINE_ACTION["place_furnace"],
    "Place Plant": MAP_SLUG_TO_ENGINE_ACTION["place_plant"],
    "Make Wood Pickaxe": MAP_SLUG_TO_ENGINE_ACTION["make_wood_pickaxe"],
    "Make Stone Pickaxe": MAP_SLUG_TO_ENGINE_ACTION["make_stone_pickaxe"],
    "Make Iron Pickaxe": MAP_SLUG_TO_ENGINE_ACTION["make_iron_pickaxe"],
    "Make Wood Sword": MAP_SLUG_TO_ENGINE_ACTION["make_wood_sword"],
    "Make Stone Sword": MAP_SLUG_TO_ENGINE_ACTION["make_stone_sword"],
    "Make Iron Sword": MAP_SLUG_TO_ENGINE_ACTION["make_iron_sword"],
}


@dataclass
class TextRendererOutput:
    long_term_context: str
    short_term_context: str


class TextRendererProtocol(Protocol):
    def __call__(self, world_state: WorldState) -> TextRendererOutput: ...


def _manhattan(dx: int, dy: int) -> int:
    return abs(dx) + abs(dy)


def _direction_words(dx: int, dy: int) -> str:
    parts: list[str] = []
    if dy < 0:
        parts.append(f"{abs(dy)} north")
    elif dy > 0:
        parts.append(f"{abs(dy)} south")
    if dx < 0:
        parts.append(f"{abs(dx)} west")
    elif dx > 0:
        parts.append(f"{abs(dx)} east")
    if not parts:
        return "here"
    return ", ".join(parts)


def _facing_to_word(facing: Position) -> str:
    if facing.x < 0 and facing.y == 0:
        return "west"
    if facing.x > 0 and facing.y == 0:
        return "east"
    if facing.x == 0 and facing.y < 0:
        return "north"
    if facing.x == 0 and facing.y > 0:
        return "south"
    return "unknown"


class UnsupervisedTextRenderer:
    """
    WorldState-based text renderer for the unsupervised Crafter environment.

    - Uses only exported WorldState (no engine internals)
    - Short-term context: inventory + local view summary
    - Long-term context: status, facing/targeting, nearest-outside summary
    """

    def __init__(
        self,
        *,
        interesting_materials: Optional[set[str]] = None,
        interesting_entities: Optional[set[str]] = None,
        report_k_nearest_outside: int = 1,
        include_zero_inventory: bool = True,
    ) -> None:
        if interesting_materials is None:
            interesting_materials = {
                "water",
                "tree",
                "stone",
                "coal",
                "iron",
                "diamond",
                "table",
                "furnace",
            }
        if interesting_entities is None:
            interesting_entities = {
                "cow",
                "zombie",
                "skeleton",
                "plant",
                "fence",
                "arrow",
            }
        self.interesting_materials = interesting_materials
        self.interesting_entities = interesting_entities
        self.report_k_nearest_outside = report_k_nearest_outside
        self.include_zero_inventory = include_zero_inventory

    def __call__(self, world_state: WorldState) -> TextRendererOutput:
        player = world_state.player
        px, py = player.position.x, player.position.y

        # Inventory summary (include zeros if configured)
        inv_items = player.inventory.model_dump()
        if self.include_zero_inventory:
            inv_lines = [f"- {k}: {v}" for k, v in inv_items.items()]
        else:
            inv_lines = [f"- {k}: {v}" for k, v in inv_items.items() if v != 0]
        inventory_section = "Your inventory:\n" + (
            "\n".join(inv_lines) if inv_lines else "(empty)"
        )

        # Local view radius from update_range (Manhattan distance)
        update_range = world_state.update_range

        # Collect interesting materials within local view
        local_material_obs: list[str] = []
        size_x, size_y = world_state.size
        materials = world_state.materials
        for x in range(size_x):
            for y in range(size_y):
                mat = materials[x][y]
                if mat is None:
                    continue
                if mat not in self.interesting_materials:
                    continue
                dx, dy = x - px, y - py
                if _manhattan(dx, dy) <= update_range:
                    rel = _direction_words(dx, dy)
                    local_material_obs.append(f"- {mat} at {rel} (x={x}, y={y})")

        # Collect entities within local view (exclude player)
        local_entity_obs: list[str] = []
        for obj in world_state.objects:
            if obj is world_state.player:
                continue
            name = obj.name
            if name not in self.interesting_entities:
                # still show entities, but we prioritize interesting
                pass
            ox, oy = obj.position.x, obj.position.y
            dx, dy = ox - px, oy - py
            if _manhattan(dx, dy) <= update_range:
                rel = _direction_words(dx, dy)
                local_entity_obs.append(f"- {name} at {rel} (x={ox}, y={oy})")

        local_view_lines: list[str] = ["Local view (within update range):"]
        if local_material_obs:
            local_view_lines.append("Materials:")
            local_view_lines.extend(local_material_obs)
        if local_entity_obs:
            local_view_lines.append("Entities:")
            local_view_lines.extend(local_entity_obs)
        if len(local_view_lines) == 1:
            local_view_lines.append("(nothing notable in local view)")
        local_view_section = "\n".join(local_view_lines)

        # Facing/targeting
        facing_word = _facing_to_word(player.facing)
        target_material, target_obj = world_state.get_target_tile()
        if target_obj is not None:
            targeting_desc = f"you are targeting a {target_obj.name}"
            # also include material under the targeted tile, if any
            if target_material is not None:
                targeting_desc += f" on {target_material}"
        else:
            patch = target_material if target_material is not None else "nothing"
            if patch == "nothing":
                targeting_desc = "you are targeting nothing"
            else:
                targeting_desc = f"you are targeting a patch of {patch}"
        # facing_section is integrated into long_term below

        # Status
        status_bits: list[str] = []
        if player.sleeping:
            status_bits.append("sleeping")
        if player.health <= 0:
            status_bits.append("dead")
        status_section = "Status: " + (", ".join(status_bits) if status_bits else "ok")

        # Outside-local-view nearest-of-each-kind
        outside_lines: list[str] = ["Interesting things outside view (nearest first):"]

        def collect_positions_for_material(kind: str) -> list[tuple[int, int, int]]:
            results: list[tuple[int, int, int]] = []
            for x in range(size_x):
                for y in range(size_y):
                    mat = materials[x][y]
                    if mat != kind:
                        continue
                    dx, dy = x - px, y - py
                    if _manhattan(dx, dy) > update_range:
                        results.append((_manhattan(dx, dy), x, y))
            results.sort(key=lambda t: (t[0], t[1], t[2]))
            return results

        def collect_positions_for_entity(kind: str) -> list[tuple[int, int, int]]:
            results: list[tuple[int, int, int]] = []
            for obj in world_state.objects:
                if obj is world_state.player:
                    continue
                if obj.name != kind:
                    continue
                x, y = obj.position.x, obj.position.y
                dx, dy = x - px, y - py
                if _manhattan(dx, dy) > update_range:
                    results.append((_manhattan(dx, dy), x, y))
            results.sort(key=lambda t: (t[0], t[1], t[2]))
            return results

        interesting_material_kinds = [
            "tree",
            "stone",
            "coal",
            "iron",
            "diamond",
            "table",
            "furnace",
        ]
        interesting_entity_kinds = [
            "cow",
            "zombie",
            "skeleton",
            "plant",
            "fence",
            "arrow",
        ]

        for kind in interesting_material_kinds:
            positions = collect_positions_for_material(kind)
            if not positions:
                outside_lines.append(f"- {kind}: not present")
                continue
            topk = positions[: self.report_k_nearest_outside]
            joined = "; ".join(f"{dist} steps at (x={x}, y={y})" for dist, x, y in topk)
            outside_lines.append(f"- {kind}: {joined}")

        for kind in interesting_entity_kinds:
            positions = collect_positions_for_entity(kind)
            if not positions:
                outside_lines.append(f"- {kind}: not present")
                continue
            topk = positions[: self.report_k_nearest_outside]
            joined = "; ".join(f"{dist} steps at (x={x}, y={y})" for dist, x, y in topk)
            outside_lines.append(f"- {kind}: {joined}")

        outside_section = "\n".join(outside_lines)

        long_term = "\n\n".join(
            [status_section, f"facing: {facing_word}", targeting_desc, outside_section]
        )
        short_term = "\n\n".join([inventory_section, local_view_section])
        return TextRendererOutput(
            long_term_context=long_term.strip(), short_term_context=short_term.strip()
        )


class UnsupervisedCrafterEnvironmentConfig(EnvironmentConfig):
    area: tuple[int, int]
    view: tuple[int, int]
    size: tuple[int, int]
    text_renderer: Callable[[WorldState], TextRendererOutput] = Field(
        default_factory=lambda: UnsupervisedTextRenderer()
    )
    instruction_prompt: str
    reward: bool
    seed: Optional[int] = None
    name: str = "crafter"
    task: str = "open_ended"
    max_episode_steps: int = Field(default=2000)
    render_image: bool = Field(default=False)

    # Pydantic v2 configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)


def build_base_environment(
    config: UnsupervisedCrafterEnvironmentConfig,
) -> BaseCrafterEnv:
    return BaseCrafterEnv(
        area=config.area,
        view=config.view,
        size=config.size,
        reward=config.reward,
        length=config.max_episode_steps,
        seed=config.seed,
    )


class LanguageSymbolicWrapper:
    def __init__(self, config: UnsupervisedCrafterEnvironmentConfig):
        self.config = config
        self.base_env = build_base_environment(config)
        self.renderer = config.text_renderer
        self.step_count = 0
        self.score_tracker = 0
        self.achievements = None
        self.default_action = "Noop"
        self.failed_candidates: list[str] = []

    def _update_progress(self, info: dict):
        self.score_tracker = 0 + sum(
            [1.0 for k, v in info["achievements"].items() if v > 0]
        )
        self.achievements = info["achievements"]
        return self.score_tracker

    def _step(self, action: str) -> tuple[
        np.ndarray,
        float,
        bool,
        bool,
        dict,
    ]:
        obs, reward, done, info = self.base_env.step(
            MAP_DISPLAY_ACTION_TO_ENGINE_ACTION[action]
        )
        self.step_count += 1
        truncated = self.step_count >= self.config.max_episode_steps
        if truncated:
            done = True

        return obs, reward, bool(done), truncated, info

    def _build_observation(
        self, obs: np.ndarray, info: dict, world_state: WorldState
    ) -> Observation:
        language_observation = self.renderer(world_state)
        return Observation(
            text=Text(
                short_term_context=language_observation.short_term_context,
                long_term_context=language_observation.long_term_context,
            ),
            image=(
                Image.fromarray(self.base_env.render()).convert("RGB")
                if self.config.render_image
                else None
            ),
            obs=obs,
        )

    def reset(self, seed: Optional[int] = None) -> OnResetExperience[WorldState]:
        """Reset the environment and return both language and symbolic observations."""
        # Crafter's `reset` method does not accept a seed, so we set it after init.
        self.base_env.reset()
        self.base_env._seed = seed
        self.step_count = 0
        self.score_tracker = 0
        self.achievements = None

        assert self.base_env._step is not None
        assert self.base_env._view is not None

        # Create language observation
        # One problem is that Crafter is an old-style env that does not return an
        # info dict upon calling `reset()`
        # We _could_ use the exported state here, but for consistentcy with Balrog, we will take
        # a no-op action and use the info dict from the step call.
        obs, _, _, _, info = self._step(self.default_action)

        world_state = export_world_state(
            self.base_env._world,
            tuple(self.base_env._view),
            self.base_env._step,
        )

        language_observation = self.renderer(world_state)
        language_obs = Observation(
            text=Text(
                short_term_context=language_observation.short_term_context,
                long_term_context=language_observation.long_term_context,
            ),
            image=(
                Image.fromarray(self.base_env.render()).convert("RGB")
                if self.config.render_image
                else None
            ),
            obs=obs,
        )

        return OnResetExperience(
            obs=language_obs,
            info=world_state,
        )

    def step(self, action: str) -> Experience[WorldState]:
        obs, reward, done, truncated, info = self._step(action)
        assert self.base_env._step is not None
        world_state = export_world_state(
            self.base_env._world,
            tuple(self.base_env._view),
            self.base_env._step,
        )
        language_obs = self._build_observation(obs, info, world_state)
        self.score_tracker = self._update_progress(info)
        return Experience(
            obs=language_obs,
            reward=reward,
            done=done,
            action=action,
            truncated=truncated,
            info=world_state,
        )

    def get_stats(self) -> dict:
        return {
            "score": self.score_tracker,
            "progression": float(self.score_tracker) / 22.0,
            "achievements": self.achievements,
        }

    def check_action_validity(self, candidate_action: str) -> str:
        if candidate_action in MAP_DISPLAY_ACTION_TO_ENGINE_ACTION:
            return candidate_action
        else:
            self.failed_candidates.append(candidate_action)
            return self.default_action

    def get_instruction_prompt(self, instructions: str | None = None) -> str:
        return self.config.instruction_prompt


implements(EnvironmentProtocol[WorldState])(LanguageSymbolicWrapper)
