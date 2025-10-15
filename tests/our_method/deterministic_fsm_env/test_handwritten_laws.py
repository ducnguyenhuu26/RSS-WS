from onelife.our_method.deterministic_fsm_env.handwritten_laws import (
    CorrectToggleALaw,
    CorrectToggleBLaw,
    IncorrectToggleALawStaysSame,
    IncorrectToggleBLawStaysSame,
    IncorrectToggleALawTogglesB,
    IncorrectToggleBLawTogglesA,
)
from onelife.deterministic_fsm_env import State, Action
from onelife.poe_world.core import DiscreteDistribution
from typing import cast


def test_correct_toggle_a_expert():
    state = State(switch_a=0, switch_b=1)
    action = Action.TOGGLE_A
    CorrectToggleALaw().effect(state, action)
    prediction = cast(DiscreteDistribution, state.switch_a)
    assert prediction.support[0] == 1


def test_correct_toggle_b_expert():
    state = State(switch_a=0, switch_b=1)
    action = Action.TOGGLE_B
    CorrectToggleBLaw().effect(state, action)
    prediction = cast(DiscreteDistribution, state.switch_b)
    assert prediction.support[0] == 0


def test_incorrect_toggle_a_expert_stays_same():
    state = State(switch_a=0, switch_b=1)
    action = Action.TOGGLE_A
    IncorrectToggleALawStaysSame().effect(state, action)
    prediction = cast(DiscreteDistribution, state.switch_a)
    assert prediction.support[0] == 0


def test_incorrect_toggle_b_expert_stays_same():
    state = State(switch_a=0, switch_b=1)
    action = Action.TOGGLE_B
    IncorrectToggleBLawStaysSame().effect(state, action)
    prediction = cast(DiscreteDistribution, state.switch_b)
    assert prediction.support[0] == 1


def test_incorrect_toggle_a_expert_toggles_b():
    state = State(switch_a=0, switch_b=1)
    action = Action.TOGGLE_A
    IncorrectToggleALawTogglesB().effect(state, action)
    prediction = cast(DiscreteDistribution, state.switch_b)
    assert prediction.support[0] == 0


def test_incorrect_toggle_b_expert_toggles_a():
    state = State(switch_a=0, switch_b=1)
    action = Action.TOGGLE_B
    IncorrectToggleBLawTogglesA().effect(state, action)
    prediction = cast(DiscreteDistribution, state.switch_a)
    assert prediction.support[0] == 1
