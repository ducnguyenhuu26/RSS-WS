from .interface import Mutator
from ....typing_utils import implements
from crafter.state_export import WorldState
from crafter.constants import ActionT, CollectableT, collect
import random
from typing import cast
from crafter.functional_env import transition
from distant_sunburn.evaluator.crafter.utils import MAP_ACTION_TO_INDEX
from loguru import logger
from typing_extensions import assert_never
from crafter.constants import MaterialT, materials

TILE_TO_MATERIALS: dict[MaterialT, CollectableT] = {
    "tree": "wood",
    "stone": "stone",
    "coal": "coal",
    "iron": "iron",
    "diamond": "diamond",
    "water": "drink",
    "grass": "sapling",
}

HARVESTABLE_TILES: set[MaterialT] = set(TILE_TO_MATERIALS.keys())


class CollectIllegalMaterialMutator:
    def __init__(self):
        self.category = "Collection"
        self.logger = logger.bind(mutator=self.__class__.__name__)

    def precondition(self, state: WorldState, action: ActionT) -> bool:
        # We make a choice here to treat any "do" action as a collection action.
        # This is not always the case since "do" is contextual to the facing
        # tile. For example, is the player is facing a entity, then the
        # do action might result in combat with the entity.
        return action == "do"

    def __call__(self, state: WorldState, action: ActionT) -> WorldState:
        mutated_state = state.model_copy(deep=True)

        # The thing we have to do here is to make sure we are not
        # collecting a material that the player _should_ be able to
        # collect in the true next state.

        # First, we check what the player is facing.
        material, _ = state.get_target_tile()

        if material is None:
            # We're facing the edge of the world, thus anything we
            # can collect is illegal.
            unobtainable_tile = random.choice(list(HARVESTABLE_TILES))
            illegal_collectable = TILE_TO_MATERIALS[unobtainable_tile]
        else:
            # We have a material! We will now pretend like the tile we are
            # facing is actually one of the _other_ materials, specifically,
            # a tile material that is harvestable.

            disjunction = HARVESTABLE_TILES - cast(set[MaterialT], {material})

            unobtainable_tile = random.choice(list(disjunction))

            illegal_collectable = TILE_TO_MATERIALS[unobtainable_tile]

        self.logger.debug(
            f"Pretending {material} is actually {unobtainable_tile} and collecting {illegal_collectable}"
        )

        # Now we increment the player's inventory by 1 of the illegal collectable.
        match illegal_collectable:
            case "wood":
                mutated_state.player.inventory.wood += 1
            case "stone":
                mutated_state.player.inventory.stone += 1
            case "coal":
                mutated_state.player.inventory.coal += 1
            case "iron":
                mutated_state.player.inventory.iron += 1
            case "diamond":
                mutated_state.player.inventory.diamond += 1
            case "drink":
                mutated_state.player.inventory.drink += 1
            case "sapling":
                mutated_state.player.inventory.sapling += 1
            case "grass":
                mutated_state.player.inventory.sapling += 1
            case _:
                logger.warning(
                    f"Received a collectable {illegal_collectable} that cannot be added to the inventory; "
                    f"this mutator should only be applied to collectable materials. "
                    f".precondition() returns {self.precondition(state, action)} for this (s,a) pair."
                    f" Setting all items to -1 to create a guaranteed illegal state."
                )
                mutated_state.player.inventory.wood = -1
                mutated_state.player.inventory.stone = -1
                mutated_state.player.inventory.coal = -1
                mutated_state.player.inventory.iron = -1
                mutated_state.player.inventory.diamond = -1
                mutated_state.player.inventory.drink = -1
                mutated_state.player.inventory.sapling = -1

        return mutated_state


implements(Mutator)(CollectIllegalMaterialMutator)
