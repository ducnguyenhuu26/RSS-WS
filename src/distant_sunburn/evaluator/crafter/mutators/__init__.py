from .collection import CollectIllegalMaterialMutator
from .crafting import CraftIllegalItemMutator
from .movement import IllegalMovementMutator
from .entity_position import EntityPositionMutator
from .interface import Mutator
from .entity_health import PlayerHealthMutator
from .player import InventoryMutator

DEFAULT_MUTATORS = [
    CollectIllegalMaterialMutator(),
    CraftIllegalItemMutator(),
    IllegalMovementMutator(),
    EntityPositionMutator(),
    PlayerHealthMutator(),
    InventoryMutator(),
]
