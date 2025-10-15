from ..core import ExpertFunctionWrapper, DiscreteDistribution
from ...mixed_fsm_env import State, Action


def correct_deterministic_switch_expert(state: State, action: Action) -> None:
    if action == Action.TOGGLE_DETERMINISTIC_SWITCH:
        state.deterministic_switch = DiscreteDistribution(support=[int(not state.deterministic_switch)])  # type: ignore


def correct_stochastic_switch_expert(state: State, action: Action) -> None:
    if action == Action.TOGGLE_STOCHASTIC_SWITCH:
        state.stochastic_switch = DiscreteDistribution(support=[0, 1])  # type: ignore


def correct_static_switch_expert(state: State, action: Action) -> None:
    if action == Action.TOGGLE_STATIC_SWITCH:
        state.static_switch = DiscreteDistribution(support=[state.static_switch])  # type: ignore


def incorrect_deterministic_switch_expert_assumes_static(
    state: State, action: Action
) -> None:
    # Assumes the deterministic switch is static
    if action == Action.TOGGLE_DETERMINISTIC_SWITCH:
        state.deterministic_switch = DiscreteDistribution(support=[state.deterministic_switch])  # type: ignore


def incorrect_deterministic_switch_expert_assumes_stochastic(
    state: State, action: Action
) -> None:
    # Assumes the deterministic switch is stochastic
    if action == Action.TOGGLE_DETERMINISTIC_SWITCH:
        state.deterministic_switch = DiscreteDistribution(support=[0, 1])  # type: ignore


def incorrect_stochastic_switch_expert_assumes_static(
    state: State, action: Action
) -> None:
    # Assumes the stochastic switch is static
    if action == Action.TOGGLE_STOCHASTIC_SWITCH:
        state.stochastic_switch = DiscreteDistribution(support=[state.stochastic_switch])  # type: ignore


def incorrect_stochastic_switch_expert_assumes_deterministic(
    state: State, action: Action
) -> None:
    # Assumes the stochastic switch is deterministic
    if action == Action.TOGGLE_STOCHASTIC_SWITCH:
        state.stochastic_switch = DiscreteDistribution(support=[int(not state.stochastic_switch)])  # type: ignore


def incorrect_static_switch_expert_assumes_deterministic(
    state: State, action: Action
) -> None:
    # Assumes the static switch is deterministic
    if action == Action.TOGGLE_STATIC_SWITCH:
        state.static_switch = DiscreteDistribution(support=[int(not state.static_switch)])  # type: ignore


def incorrect_static_switch_expert_assumes_stochastic(
    state: State, action: Action
) -> None:
    # Assumes the static switch is stochastic
    if action == Action.TOGGLE_STATIC_SWITCH:
        state.static_switch = DiscreteDistribution(support=[0, 1])  # type: ignore


CORRECT_EXPERTS = [
    ExpertFunctionWrapper.from_non_runtime_created(correct_deterministic_switch_expert),
    ExpertFunctionWrapper.from_non_runtime_created(correct_stochastic_switch_expert),
    ExpertFunctionWrapper.from_non_runtime_created(correct_static_switch_expert),
]

INCORRECT_EXPERTS = [
    ExpertFunctionWrapper.from_non_runtime_created(
        incorrect_deterministic_switch_expert_assumes_static
    ),
    ExpertFunctionWrapper.from_non_runtime_created(
        incorrect_deterministic_switch_expert_assumes_stochastic
    ),
    ExpertFunctionWrapper.from_non_runtime_created(
        incorrect_stochastic_switch_expert_assumes_static
    ),
    ExpertFunctionWrapper.from_non_runtime_created(
        incorrect_stochastic_switch_expert_assumes_deterministic
    ),
    ExpertFunctionWrapper.from_non_runtime_created(
        incorrect_static_switch_expert_assumes_deterministic
    ),
    ExpertFunctionWrapper.from_non_runtime_created(
        incorrect_static_switch_expert_assumes_stochastic
    ),
]

ALL_EXPERTS = CORRECT_EXPERTS + INCORRECT_EXPERTS
