from onelife.poe_world.deterministic_fsm_env.handwritten_experts import (
    correct_toggle_a_expert,
    correct_toggle_b_expert,
    incorrect_toggle_a_expert_stays_same,
    incorrect_toggle_b_expert_stays_same,
    incorrect_toggle_a_expert_toggles_b,
    incorrect_toggle_b_expert_toggles_a,
)
from onelife.deterministic_fsm_env import (
    State,
    Action,
)
from onelife.poe_world.core import DiscreteDistribution
from typing import cast


def test_correct_toggle_a_expert():
    state = State(switch_a=0, switch_b=1)
    action = Action.TOGGLE_A
    correct_toggle_a_expert(state, action)
    prediction = cast(DiscreteDistribution, state.switch_a)
    assert prediction.support[0] == 1


def test_correct_toggle_b_expert():
    state = State(switch_a=0, switch_b=1)
    action = Action.TOGGLE_B
    correct_toggle_b_expert(state, action)
    prediction = cast(DiscreteDistribution, state.switch_b)
    assert prediction.support[0] == 0


def test_incorrect_toggle_a_expert_stays_same():
    state = State(switch_a=0, switch_b=1)
    action = Action.TOGGLE_A
    incorrect_toggle_a_expert_stays_same(state, action)
    prediction = cast(DiscreteDistribution, state.switch_a)
    assert prediction.support[0] == 0


def test_incorrect_toggle_b_expert_stays_same():
    state = State(switch_a=0, switch_b=1)
    action = Action.TOGGLE_B
    incorrect_toggle_b_expert_stays_same(state, action)
    prediction = cast(DiscreteDistribution, state.switch_b)
    assert prediction.support[0] == 1


def test_incorrect_toggle_a_expert_toggles_b():
    state = State(switch_a=0, switch_b=1)
    action = Action.TOGGLE_A
    incorrect_toggle_a_expert_toggles_b(state, action)
    prediction = cast(DiscreteDistribution, state.switch_b)
    assert prediction.support[0] == 0


def test_incorrect_toggle_b_expert_toggles_a():
    state = State(switch_a=0, switch_b=1)
    action = Action.TOGGLE_B
    incorrect_toggle_b_expert_toggles_a(state, action)
    prediction = cast(DiscreteDistribution, state.switch_a)
    assert prediction.support[0] == 1
