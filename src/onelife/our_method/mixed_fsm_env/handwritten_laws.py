from ...poe_world.core import DiscreteDistribution
from ...mixed_fsm_env import State, Action
from ..core import LawFunctionWrapper


class CorrectDeterministicSwitchLaw:
    def precondition(self, current_state: State, action: Action) -> bool:
        return action == Action.TOGGLE_DETERMINISTIC_SWITCH

    def effect(self, current_state: State, action: Action) -> None:
        current_state.deterministic_switch = DiscreteDistribution(support=[int(not current_state.deterministic_switch)])  # type: ignore


class CorrectStochasticSwitchLaw:
    def precondition(self, current_state: State, action: Action) -> bool:
        return action == Action.TOGGLE_STOCHASTIC_SWITCH

    def effect(self, current_state: State, action: Action) -> None:
        current_state.stochastic_switch = DiscreteDistribution(support=[0, 1])  # type: ignore


class CorrectStaticSwitchLaw:
    def precondition(self, current_state: State, action: Action) -> bool:
        return action == Action.TOGGLE_STATIC_SWITCH

    def effect(self, current_state: State, action: Action) -> None:
        current_state.static_switch = DiscreteDistribution(support=[current_state.static_switch])  # type: ignore


class IncorrectDeterministicSwitchLawAssumesStatic:
    def precondition(self, current_state: State, action: Action) -> bool:
        return action == Action.TOGGLE_DETERMINISTIC_SWITCH

    def effect(self, current_state: State, action: Action) -> None:
        # Assumes the deterministic switch is static
        current_state.deterministic_switch = DiscreteDistribution(support=[current_state.deterministic_switch])  # type: ignore


class IncorrectDeterministicSwitchLawAssumesStochastic:
    def precondition(self, current_state: State, action: Action) -> bool:
        return action == Action.TOGGLE_DETERMINISTIC_SWITCH

    def effect(self, current_state: State, action: Action) -> None:
        # Assumes the deterministic switch is stochastic
        current_state.deterministic_switch = DiscreteDistribution(support=[0, 1])  # type: ignore


class IncorrectStochasticSwitchLawAssumesStatic:
    def precondition(self, current_state: State, action: Action) -> bool:
        return action == Action.TOGGLE_STOCHASTIC_SWITCH

    def effect(self, current_state: State, action: Action) -> None:
        # Assumes the stochastic switch is static
        current_state.stochastic_switch = DiscreteDistribution(support=[current_state.stochastic_switch])  # type: ignore


class IncorrectStochasticSwitchLawAssumesDeterministic:
    def precondition(self, current_state: State, action: Action) -> bool:
        return action == Action.TOGGLE_STOCHASTIC_SWITCH

    def effect(self, current_state: State, action: Action) -> None:
        # Assumes the stochastic switch is deterministic
        current_state.stochastic_switch = DiscreteDistribution(support=[int(not current_state.stochastic_switch)])  # type: ignore


class IncorrectStaticSwitchLawAssumesDeterministic:
    def precondition(self, current_state: State, action: Action) -> bool:
        return action == Action.TOGGLE_STATIC_SWITCH

    def effect(self, current_state: State, action: Action) -> None:
        # Assumes the static switch is deterministic
        current_state.static_switch = DiscreteDistribution(support=[int(not current_state.static_switch)])  # type: ignore


class IncorrectStaticSwitchLawAssumesStochastic:
    def precondition(self, current_state: State, action: Action) -> bool:
        return action == Action.TOGGLE_STATIC_SWITCH

    def effect(self, current_state: State, action: Action) -> None:
        # Assumes the static switch is stochastic
        current_state.static_switch = DiscreteDistribution(support=[0, 1])  # type: ignore


CORRECT_LAWS = [
    LawFunctionWrapper[State].from_non_runtime_created(CorrectDeterministicSwitchLaw()),
    LawFunctionWrapper[State].from_non_runtime_created(CorrectStochasticSwitchLaw()),
    LawFunctionWrapper[State].from_non_runtime_created(CorrectStaticSwitchLaw()),
]

INCORRECT_LAWS = [
    LawFunctionWrapper[State].from_non_runtime_created(
        IncorrectDeterministicSwitchLawAssumesStatic()
    ),
    LawFunctionWrapper[State].from_non_runtime_created(
        IncorrectDeterministicSwitchLawAssumesStochastic()
    ),
    LawFunctionWrapper[State].from_non_runtime_created(
        IncorrectStochasticSwitchLawAssumesStatic()
    ),
    LawFunctionWrapper[State].from_non_runtime_created(
        IncorrectStochasticSwitchLawAssumesDeterministic()
    ),
    LawFunctionWrapper[State].from_non_runtime_created(
        IncorrectStaticSwitchLawAssumesDeterministic()
    ),
    LawFunctionWrapper[State].from_non_runtime_created(
        IncorrectStaticSwitchLawAssumesStochastic()
    ),
]

ALL_LAWS = CORRECT_LAWS + INCORRECT_LAWS
