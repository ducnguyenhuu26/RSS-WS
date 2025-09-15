from ..core import LawFunctionWrapper
from ...deterministic_fsm_env import State, Action
from ...poe_world.core import DiscreteDistribution


class CorrectToggleALaw:
    def precondition(self, current_state: State, action: Action) -> bool:
        return action == Action.TOGGLE_A

    def effect(self, current_state: State, action: Action) -> None:
        current_state.switch_a = DiscreteDistribution(support=[int(not current_state.switch_a)])  # type: ignore


class CorrectToggleBLaw:
    def precondition(self, current_state: State, action: Action) -> bool:
        return action == Action.TOGGLE_B

    def effect(self, current_state: State, action: Action) -> None:
        current_state.switch_b = DiscreteDistribution(support=[int(not current_state.switch_b)])  # type: ignore


class IncorrectToggleALawStaysSame:
    def precondition(self, current_state: State, action: Action) -> bool:
        return action == Action.TOGGLE_A

    def effect(self, current_state: State, action: Action) -> None:
        current_state.switch_a = DiscreteDistribution(support=[current_state.switch_a])  # type: ignore


class IncorrectToggleBLawStaysSame:
    def precondition(self, current_state: State, action: Action) -> bool:
        return action == Action.TOGGLE_B

    def effect(self, current_state: State, action: Action) -> None:
        current_state.switch_b = DiscreteDistribution(support=[current_state.switch_b])  # type: ignore


class IncorrectToggleALawTogglesB:
    def precondition(self, current_state: State, action: Action) -> bool:
        return action == Action.TOGGLE_A

    def effect(self, current_state: State, action: Action) -> None:
        current_state.switch_b = DiscreteDistribution(support=[int(not current_state.switch_b)])  # type: ignore


class IncorrectToggleBLawTogglesA:
    def precondition(self, current_state: State, action: Action) -> bool:
        return action == Action.TOGGLE_B

    def effect(self, current_state: State, action: Action) -> None:
        current_state.switch_a = DiscreteDistribution(support=[int(not current_state.switch_a)])  # type: ignore


CORRECT_LAWS = [
    LawFunctionWrapper[State].from_non_runtime_created(CorrectToggleALaw()),
    LawFunctionWrapper[State].from_non_runtime_created(CorrectToggleBLaw()),
]

INCORRECT_LAWS = [
    LawFunctionWrapper[State].from_non_runtime_created(IncorrectToggleALawStaysSame()),
    LawFunctionWrapper[State].from_non_runtime_created(IncorrectToggleBLawStaysSame()),
    LawFunctionWrapper[State].from_non_runtime_created(IncorrectToggleALawTogglesB()),
    LawFunctionWrapper[State].from_non_runtime_created(IncorrectToggleBLawTogglesA()),
]

ALL_LAWS = CORRECT_LAWS + INCORRECT_LAWS
