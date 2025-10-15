from ..core import ExpertFunction, ExpertFunctionWrapper, DiscreteDistribution
from ...deterministic_fsm_env import State, Action


def correct_toggle_a_expert(current_state: State, action: Action) -> None:
    if action == Action.TOGGLE_A:
        value = int(not current_state.switch_a)
        current_state.switch_a = DiscreteDistribution(support=[value])  # type: ignore


def correct_toggle_b_expert(current_state: State, action: Action) -> None:
    if action == Action.TOGGLE_B:
        value = int(not current_state.switch_b)
        current_state.switch_b = DiscreteDistribution(support=[value])  # type: ignore


def incorrect_toggle_a_expert_stays_same(current_state: State, action: Action) -> None:
    if action == Action.TOGGLE_A:
        current_state.switch_a = DiscreteDistribution(support=[current_state.switch_a])  # type: ignore


def incorrect_toggle_b_expert_stays_same(current_state: State, action: Action) -> None:
    if action == Action.TOGGLE_B:
        current_state.switch_b = DiscreteDistribution(support=[current_state.switch_b])  # type: ignore


def incorrect_toggle_a_expert_toggles_b(current_state: State, action: Action) -> None:
    if action == Action.TOGGLE_A:
        current_state.switch_b = DiscreteDistribution(support=[int(not current_state.switch_b)])  # type: ignore


def incorrect_toggle_b_expert_toggles_a(current_state: State, action: Action) -> None:
    if action == Action.TOGGLE_B:
        current_state.switch_a = DiscreteDistribution(support=[int(not current_state.switch_a)])  # type: ignore


CORRECT_EXPERTS = [
    ExpertFunctionWrapper[State].from_non_runtime_created(correct_toggle_a_expert),
    ExpertFunctionWrapper[State].from_non_runtime_created(correct_toggle_b_expert),
]

INCORRECT_EXPERTS = [
    ExpertFunctionWrapper[State].from_non_runtime_created(
        incorrect_toggle_a_expert_stays_same
    ),
    ExpertFunctionWrapper[State].from_non_runtime_created(
        incorrect_toggle_b_expert_stays_same
    ),
    ExpertFunctionWrapper[State].from_non_runtime_created(
        incorrect_toggle_a_expert_toggles_b
    ),
    ExpertFunctionWrapper[State].from_non_runtime_created(
        incorrect_toggle_b_expert_toggles_a
    ),
]

ALL_EXPERTS = CORRECT_EXPERTS + INCORRECT_EXPERTS
