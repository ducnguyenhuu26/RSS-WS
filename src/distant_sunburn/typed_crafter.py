from dataclasses import dataclass
import numpy as np
from typing import Optional, SupportsFloat, Any

from crafter.env import Env as CrafterBaseEnv
from gymnasium.core import Env
from gymnasium import spaces
import numpy.typing as npt


# 1. Crafter configuration dataclass
@dataclass
class CrafterConfig:
    area: tuple[int, int] = (64, 64)
    view: tuple[int, int] = (9, 9)
    size: tuple[int, int] = (64, 64)
    reward: bool = True
    length: int = 10000
    seed: Optional[int] = None


# 2. Inventory dataclass (fields match data.yaml/constants.py)
@dataclass
class Inventory:
    health: int
    food: int
    drink: int
    energy: int
    sapling: int
    wood: int
    stone: int
    coal: int
    iron: int
    diamond: int
    wood_pickaxe: int
    stone_pickaxe: int
    iron_pickaxe: int
    wood_sword: int
    stone_sword: int
    iron_sword: int

    @staticmethod
    def from_dict(d: dict) -> "Inventory":
        return Inventory(
            health=int(d["health"]),
            food=int(d["food"]),
            drink=int(d["drink"]),
            energy=int(d["energy"]),
            sapling=int(d["sapling"]),
            wood=int(d["wood"]),
            stone=int(d["stone"]),
            coal=int(d["coal"]),
            iron=int(d["iron"]),
            diamond=int(d["diamond"]),
            wood_pickaxe=int(d["wood_pickaxe"]),
            stone_pickaxe=int(d["stone_pickaxe"]),
            iron_pickaxe=int(d["iron_pickaxe"]),
            wood_sword=int(d["wood_sword"]),
            stone_sword=int(d["stone_sword"]),
            iron_sword=int(d["iron_sword"]),
        )


# 3. Achievements dataclass (fields match data.yaml)
@dataclass
class Achievements:
    collect_coal: int
    collect_diamond: int
    collect_drink: int
    collect_iron: int
    collect_sapling: int
    collect_stone: int
    collect_wood: int
    defeat_skeleton: int
    defeat_zombie: int
    eat_cow: int
    eat_plant: int
    make_iron_pickaxe: int
    make_iron_sword: int
    make_stone_pickaxe: int
    make_stone_sword: int
    make_wood_pickaxe: int
    make_wood_sword: int
    place_furnace: int
    place_plant: int
    place_stone: int
    place_table: int
    wake_up: int

    @staticmethod
    def from_dict(d: dict) -> "Achievements":
        return Achievements(
            collect_coal=int(d["collect_coal"]),
            collect_diamond=int(d["collect_diamond"]),
            collect_drink=int(d["collect_drink"]),
            collect_iron=int(d["collect_iron"]),
            collect_sapling=int(d["collect_sapling"]),
            collect_stone=int(d["collect_stone"]),
            collect_wood=int(d["collect_wood"]),
            defeat_skeleton=int(d["defeat_skeleton"]),
            defeat_zombie=int(d["defeat_zombie"]),
            eat_cow=int(d["eat_cow"]),
            eat_plant=int(d["eat_plant"]),
            make_iron_pickaxe=int(d["make_iron_pickaxe"]),
            make_iron_sword=int(d["make_iron_sword"]),
            make_stone_pickaxe=int(d["make_stone_pickaxe"]),
            make_stone_sword=int(d["make_stone_sword"]),
            make_wood_pickaxe=int(d["make_wood_pickaxe"]),
            make_wood_sword=int(d["make_wood_sword"]),
            place_furnace=int(d["place_furnace"]),
            place_plant=int(d["place_plant"]),
            place_stone=int(d["place_stone"]),
            place_table=int(d["place_table"]),
            wake_up=int(d["wake_up"]),
        )


# 4. Step info dataclass
@dataclass
class CrafterStepInfo:
    inventory: Inventory
    achievements: Achievements
    discount: float
    semantic: np.ndarray  # int values, shape (area_x, area_y), dtype usually uint8
    player_pos: tuple[int, int]
    reward: float
    truncated: bool = False
    terminated: bool = False


# 5. The wrapper
class CrafterEnv(Env[np.ndarray, np.int64]):
    """A typed Gymnasium environment wrapper for Crafter."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, config: CrafterConfig = CrafterConfig()):
        """Initialize the Crafter environment with the given configuration."""
        super().__init__()

        self._env = CrafterBaseEnv(
            area=config.area,
            view=config.view,
            size=config.size,
            reward=config.reward,
            length=config.length,
            seed=config.seed,
        )
        self._max_steps = config.length
        self._step_count = 0

        # Set up spaces
        self.observation_space = spaces.Box(
            low=0, high=255, shape=self._env.observation_space.shape, dtype=np.uint8
        )
        self.action_space = spaces.Discrete(self._env.action_space.n)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment to an initial state."""
        super().reset(seed=seed)  # Important: call super().reset() to properly seed
        obs = self._env.reset()
        info = self._make_info()
        return np.asarray(obs), info.__dict__

    def step(
        self, action: int
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        """Run one timestep of the environment's dynamics."""
        self._step_count += 1
        obs, reward, done, info_dict = self._env.step(action)

        # Determine truncation/termination like gymnasium does
        over = self._max_steps and self._step_count >= self._max_steps
        assert self._env._player is not None
        dead = self._env._player.health <= 0

        info = CrafterStepInfo(
            inventory=Inventory.from_dict(info_dict["inventory"]),
            achievements=Achievements.from_dict(info_dict["achievements"]),
            discount=float(info_dict["discount"]),
            semantic=np.asarray(info_dict["semantic"]),
            player_pos=tuple(info_dict["player_pos"]),
            reward=float(info_dict["reward"]),
            truncated=bool(over and not dead),
            terminated=bool(dead),
        )

        return (
            np.asarray(obs),
            float(reward),
            info.terminated,
            info.truncated,
            info.__dict__,
        )

    def render(self) -> np.ndarray | None:
        """Render the environment."""
        return self._env.render()

    def close(self):
        """Clean up resources."""
        return self._env.close()

    @property
    def action_names(self) -> list[str]:
        """Return the list of action names."""
        return list(self._env.action_names)

    def _make_info(self) -> CrafterStepInfo:
        """Construct info dictionary after reset."""
        assert self._env._player is not None
        player = self._env._player
        info = CrafterStepInfo(
            inventory=Inventory.from_dict(player.inventory),
            achievements=Achievements.from_dict(player.achievements),
            discount=1.0,
            semantic=np.asarray(self._env._sem_view()),
            player_pos=tuple(player.pos),
            reward=0.0,
            truncated=False,
            terminated=False,
        )
        return info


if __name__ == "__main__":
    config = CrafterConfig(area=(64, 64), reward=False, length=100)
    env = CrafterEnv(config)
    obs, info = env.reset()
    while True:
        # For Discrete action space, we can access the size with action_space.n
        assert isinstance(env.action_space, spaces.Discrete)
        action = int(np.random.randint(env.action_space.n))  # Convert to Python int
        obs, reward, terminated, truncated, info = env.step(action)
        print(info["inventory"].food)  # Access food from info dict
        if terminated or truncated:  # Check terminated or truncated
            break
