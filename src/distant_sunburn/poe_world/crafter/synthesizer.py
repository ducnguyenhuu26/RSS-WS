from crafter.state_export import (
    WorldState,
    ZombieState,
    PlayerState,
    PlantState,
    SkeletonState,
    ArrowState,
    FenceState,
)
from crafter.constants import ActionT as CrafterAction
from ..core import SymbolicTransition
from typing import TypeAlias, Union, TypeVar
from typing import Generic, Type
from ...evaluator.crafter.utils import find_all_objects_for_type
from ..core import ExpertFunction

CrafterEntityStates: TypeAlias = Union[
    ZombieState,
    PlayerState,
    PlantState,
    SkeletonState,
    ArrowState,
    FenceState,
]

CrafterEntityStateT = TypeVar("CrafterEntityStateT", bound=CrafterEntityStates)
