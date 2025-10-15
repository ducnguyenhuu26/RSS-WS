from .collection import CollectIllegalMaterialMutator
from .crafting import CraftIllegalItemMutator
from .movement import IllegalMovementMutator
from .entity_position import EntityPositionMutator
from .interface import Mutator
from .entity_health import PlayerHealthMutator, EntityHealthMutator
from .player import InventoryMutator

ALWAYS_ON_MUTATORS = [
    EntityHealthMutator(),
    EntityPositionMutator(),
    PlayerHealthMutator(),
    InventoryMutator(),
]

CONDITIONAL_MUTATORS = [
    CollectIllegalMaterialMutator(),
    CraftIllegalItemMutator(),
    IllegalMovementMutator(),
]

DEFAULT_MUTATORS = ALWAYS_ON_MUTATORS + CONDITIONAL_MUTATORS
