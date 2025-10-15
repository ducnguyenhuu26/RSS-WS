from .interface import Mutator
from ....typing_utils import implements
from crafter_oo.state_export import WorldState
from crafter_oo.constants import ActionT
import random
from loguru import logger
from crafter_oo.state_export import Position


DIRECTIONS = (
    Position(x=0, y=1),
    Position(x=1, y=0),
    Position(x=0, y=-1),
    Position(x=-1, y=0),
)


class InventoryMutator:
    def __init__(self):
        self.category = "Ego"
        self.logger = logger.bind(mutator=self.__class__.__name__)

    def precondition(self, state: WorldState, action: ActionT) -> bool:
        return True

    def __call__(self, state: WorldState, action: ActionT) -> WorldState:
        mutated_state = state.model_copy(deep=True)

        mutated_state.player.inventory.food = random.randint(0, 100)
        mutated_state.player.inventory.drink = random.randint(0, 100)
        mutated_state.player.inventory.energy = random.randint(0, 100)
        mutated_state.player.inventory.sapling = random.randint(0, 100)
        mutated_state.player.inventory.wood = random.randint(0, 100)
        mutated_state.player.inventory.stone = random.randint(0, 100)
        mutated_state.player.inventory.coal = random.randint(0, 100)
        mutated_state.player.inventory.iron = random.randint(0, 100)
        mutated_state.player.inventory.diamond += random.randint(0, 100)
        mutated_state.player.inventory.wood_pickaxe = random.randint(0, 100)
        mutated_state.player.inventory.stone_pickaxe = random.randint(0, 100)
        mutated_state.player.inventory.iron_pickaxe = random.randint(0, 100)
        mutated_state.player.inventory.wood_sword = random.randint(0, 100)
        mutated_state.player.inventory.stone_sword = random.randint(0, 100)
        mutated_state.player.inventory.iron_sword = random.randint(0, 100)

        return mutated_state


implements(Mutator)(InventoryMutator)
