"""
Mutators for generating distractors in Crafter evaluation.

This module contains mutators that apply specific changes to WorldState objects
to create plausible but incorrect next states for testing world model understanding.
"""

import copy
import random
from typing import Protocol

from crafter.state_export import WorldState, CowState, Position
from crafter.constants import ActionT
from ..utils import find_all_objects_for_type, find_object_in_state


class Mutator(Protocol):
    """Protocol for mutators that modify WorldState objects."""

    category: str

    def precondition(self, state: WorldState, action: ActionT) -> bool:
        """Check if this mutator can be applied to the given state."""
        ...

    def __call__(self, state: WorldState, action: ActionT) -> WorldState:
        """Apply the mutation to a copy of the state and return the modified copy."""
        ...


# TODO: This doesn't actually guarantee illegal states. To do so, we'd
# need to make sure that the mutation is not one that would have been caused
# by the action. This should be handled by the mutation generator though —
# it can skip any mutations that result in the true transition via an
# equality check.
class AddIllegalItemMutator:
    """
    A mutator that adds an item to the player's inventory that
    should not be there. This tests if the world model understands
    inventory causality.
    """

    def __init__(self, item_name: str, amount: int):
        self.item_name = item_name
        self.amount = amount
        self.category = "Player State"

    def precondition(self, state: WorldState, action: ActionT) -> bool:
        # This mutator is general and can apply to any action.
        return True

    def __call__(self, state: WorldState, action: ActionT) -> WorldState:
        mutated_state = copy.deepcopy(state)
        current_amount = getattr(mutated_state.player.inventory, self.item_name)
        setattr(
            mutated_state.player.inventory, self.item_name, current_amount + self.amount
        )
        return mutated_state


# TODO: Name is wrong; this only teleports cows, not other entities.
class TeleportEntityToIllegalTileMutator:
    """
    Moves a random cow to a non-walkable tile, violating physics.
    """

    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.category = "Entity State"

    def precondition(self, state: WorldState, action: ActionT) -> bool:
        # This mutator is only interesting if a cow exists and there is a non-walkable tile.
        if not find_all_objects_for_type(state, CowState):
            return False

        # Find at least one non-walkable tile
        for x in range(state.size[0]):
            for y in range(state.size[1]):
                material, _ = state.get_tile(Position(x=x, y=y))
                if material not in {"grass", "path", "sand", "water", "lava"}:
                    return True
        return False

    def __call__(self, state: WorldState, action: ActionT) -> WorldState:
        mutated_state = copy.deepcopy(state)

        # Find a cow
        cows = find_all_objects_for_type(mutated_state, CowState)
        target_cow = self.rng.choice(cows)

        # Find an illegal tile
        illegal_tiles = []
        for x in range(mutated_state.size[0]):
            for y in range(mutated_state.size[1]):
                material, occupant = mutated_state.get_tile(Position(x=x, y=y))
                if (
                    material not in {"grass", "path", "sand", "water", "lava"}
                    and occupant is None
                ):
                    illegal_tiles.append(Position(x=x, y=y))

        if not illegal_tiles:
            # Should not happen if precondition is checked
            return state

        # Move the cow
        new_pos = self.rng.choice(illegal_tiles)
        cow_in_state = find_object_in_state(
            mutated_state, target_cow.entity_id, CowState
        )
        if cow_in_state:
            cow_in_state.position = new_pos

        return mutated_state
