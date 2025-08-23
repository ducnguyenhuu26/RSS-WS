"""
Core interfaces and classes for the hybrid evaluation framework.

This module defines the core protocols and data structures that enable
environment-agnostic evaluation of symbolic world models.
"""

import copy
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar
import numpy as np

# Type variable for metadata types
MetadataT = TypeVar("MetadataT")


class EvaluatableWorldModel(Protocol[MetadataT]):
    """Protocol for world models that can be evaluated."""

    def sample_next_state(self, current_state: MetadataT, action: Any) -> MetadataT:
        """Generate single prediction by sampling from posterior P(s_next | s, a)"""
        ...

    def evaluate_log_probability(
        self, next_state: MetadataT, current_state: MetadataT, action: Any
    ) -> float:
        """Compute log P(next_state | current_state, action)"""
        ...


class SymbolicEnvironment(Protocol[MetadataT]):
    """Minimal protocol for symbolic environments."""

    def transition(self, state: MetadataT, action: Any) -> MetadataT:
        """True transition function: (s, a) -> s'"""
        ...


class TrajectoryCollector(Protocol[MetadataT]):
    """Protocol for collecting symbolic transitions."""

    def collect_transitions(
        self, environment: SymbolicEnvironment[MetadataT], num_transitions: int
    ) -> list["SymbolicTransition[MetadataT]"]:
        """Collect symbolic transitions using environment-specific policy"""
        ...


class EditDistanceCalculator(Protocol[MetadataT]):
    """Protocol for computing edit distances between states."""

    def compute_distance(self, state1: MetadataT, state2: MetadataT) -> int | float:
        """Compute structured edit distance between two states"""
        ...


class DistractorGenerator(Protocol[MetadataT]):
    """Protocol for generating plausible but incorrect next states."""

    def generate_distractors(
        self,
        transition: "SymbolicTransition[MetadataT]",
        all_transitions: list["SymbolicTransition[MetadataT]"],
        num_distractors: int,
    ) -> list[MetadataT]:
        """Generate plausible but incorrect next states"""
        ...


@dataclass(frozen=True)
class SymbolicTransition(Generic[MetadataT]):
    """A single symbolic transition."""

    prev_metadata: MetadataT
    action: Any
    next_metadata: MetadataT


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for evaluation runs."""

    num_transitions: int = 100
    num_distractors: int = 5
    random_seed: int = 42


@dataclass(frozen=True)
class EvaluationResults:
    """Results from a hybrid evaluation run."""

    mean_generative_error: float
    discriminative_accuracy: float
    discriminative_accuracy_by_distractor_type: dict[str, float]
    total_transitions_evaluated: int


class HybridEvaluator(Generic[MetadataT]):
    """
    Core evaluator that combines generative and discriminative tests.

    Uses dependency injection to remain environment-agnostic.
    """

    def __init__(
        self,
        config: EvaluationConfig,
        trajectory_collector: TrajectoryCollector[MetadataT],
        edit_distance_calc: EditDistanceCalculator[MetadataT],
        distractor_generator: DistractorGenerator[MetadataT],
    ):
        self.config = config
        self.trajectory_collector = trajectory_collector
        self.edit_distance_calc = edit_distance_calc
        self.distractor_generator = distractor_generator

    def evaluate(
        self,
        world_model: EvaluatableWorldModel[MetadataT],
        environment: SymbolicEnvironment[MetadataT],
    ) -> EvaluationResults:
        """Core evaluation logic - environment agnostic"""

        # 1. Collect transitions using injected collector
        transitions = self.trajectory_collector.collect_transitions(
            environment, self.config.num_transitions
        )

        generative_errors = []
        discriminative_successes = []
        distractor_type_results: dict[str, list[bool]] = defaultdict(list)

        for transition in transitions:
            # 2. Generate prediction
            pred_state = world_model.sample_next_state(
                transition.prev_metadata, transition.action
            )

            # 3. Measure generative error using injected calculator
            gen_error = self.edit_distance_calc.compute_distance(
                pred_state, transition.next_metadata
            )
            generative_errors.append(gen_error)

            # 4. Generate distractors using injected generator
            distractors = self.distractor_generator.generate_distractors(
                transition, transitions, self.config.num_distractors
            )

            # 5. Construct candidate set
            candidates = [transition.next_metadata, pred_state] + distractors

            # 6. Evaluate log probabilities
            log_probs = {
                candidate: world_model.evaluate_log_probability(
                    candidate, transition.prev_metadata, transition.action
                )
                for candidate in candidates
            }

            # 7. Check discriminative success
            max_prob = max(log_probs.values())
            true_state_prob = log_probs[transition.next_metadata]
            discriminative_successes.append(true_state_prob == max_prob)

        # Convert distractor type results to means
        distractor_type_means = {
            distractor_type: float(np.mean(results))
            for distractor_type, results in distractor_type_results.items()
        }

        return EvaluationResults(
            mean_generative_error=float(np.mean(generative_errors)),
            discriminative_accuracy=float(np.mean(discriminative_successes)),
            discriminative_accuracy_by_distractor_type=distractor_type_means,
            total_transitions_evaluated=len(transitions),
        )
