from .collection import CollectIllegalMaterialMutator
from .crafting import CraftIllegalItemMutator
from .movement import IllegalMovementMutator
from .v1 import Mutator

DEFAULT_MUTATORS = [
    CollectIllegalMaterialMutator(),
    CraftIllegalItemMutator(),
    IllegalMovementMutator(),
]

__all__ = [
    "CollectIllegalMaterialMutator",
    "CraftIllegalItemMutator",
    "IllegalMovementMutator",
    "DEFAULT_MUTATORS",
    "Mutator",
]
