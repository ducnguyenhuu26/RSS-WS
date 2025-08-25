"""
Integration tests for the Crafter evaluation framework.

These tests verify that all components work together correctly to evaluate world models.
"""

from crafter.functional_env import EnvConfig
from crafter.functional_env import transition as crafter_transition_fn
from crafter.state_export import WorldState

from distant_sunburn.evaluator.baselines import NullWorldModel, TrueTransitionWorldModel
from distant_sunburn.evaluator.core import EvaluationConfig, Evaluator
from distant_sunburn.evaluator.crafter.components import _gamestate_to_json
from distant_sunburn.evaluator.crafter.factory import CrafterEvaluationFactory
from distant_sunburn.evaluator.crafter.utils import MAP_ACTION_TO_INDEX


def test_evaluation_framework_can_evaluate_world_model():
    """
    Test that the evaluation framework can successfully evaluate a world model
    and produce meaningful results using the factory and actual evaluator.
    """
    env_config = EnvConfig(size=(9, 9), view=(9, 9))
    factory = CrafterEvaluationFactory(env_config, policy_seed=42)
    config = EvaluationConfig(num_distractors=2)

    context = factory.create_context(config, num_transitions_per_scenario=30)

    def states_equal(state1: WorldState, state2: WorldState) -> bool:
        return _gamestate_to_json(state1) == _gamestate_to_json(state2)

    def crafter_transition_wrapper(state: WorldState, action) -> WorldState:
        next_state, _ = crafter_transition_fn(state, MAP_ACTION_TO_INDEX[action])
        return next_state

    true_model = TrueTransitionWorldModel(crafter_transition_wrapper, states_equal)
    null_model = NullWorldModel(states_equal)

    evaluator = Evaluator(context)

    true_results = evaluator.evaluate(true_model)

    null_results = evaluator.evaluate(null_model)

    # True model should have perfect discriminative accuracy
    assert (
        true_results.discriminative_accuracy == 1.0
    ), "True transition model should have perfect discriminative accuracy"

    #
    assert (
        true_results.mean_generative_error == 0.0
    ), "True transition model should have low generative error"

    # Null model should perform worse than true model
    assert (
        null_results.discriminative_accuracy < true_results.discriminative_accuracy
    ), "Null model should perform worse than true model"

    print(null_results.discriminative_accuracy)
    print(null_results.mean_generative_error)
    print(true_results.discriminative_accuracy)
    print(true_results.mean_generative_error)
