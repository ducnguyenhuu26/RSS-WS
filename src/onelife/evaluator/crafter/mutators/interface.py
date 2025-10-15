"""
Mutators for generating distractors in Crafter evaluation.

This module contains mutators that apply specific changes to WorldState objects
to create plausible but incorrect next states for testing world model understanding.
"""

from typing import Protocol

from crafter_oo.state_export import WorldState
from crafter_oo.constants import ActionT


class Mutator(Protocol):
    """Protocol for mutators that modify WorldState objects."""

    category: str

    def precondition(self, state: WorldState, action: ActionT) -> bool:
        """Check if this mutator can be applied to the given state."""
        ...

    def __call__(self, state: WorldState, action: ActionT) -> WorldState:
        """Apply the mutation to a copy of the state and return the modified copy."""
        ...
