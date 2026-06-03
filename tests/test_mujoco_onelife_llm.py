from __future__ import annotations

from types import SimpleNamespace

import math
import numpy as np
import pytest

from onelife.litellm_utils import OpenAILiteLlmParams
from onelife.mujoco_dataset import MuJoCoTransitions
from onelife.mujoco_onelife_llm import (
    LLMOneLifeMuJoCoSynthesizer,
    LLMOneLifeSynthesisConfig,
    compile_onelife_mujoco_laws,
    evaluate_onelife_llm_baseline,
    extract_python_code,
)
from onelife.mujoco_symbolic_adapter import MuJoCoDiscretizer


GENERATED_CODE = """
```python
class ActionMovesFirstStateBin(BinnedMuJoCoLaw):
    def __init__(self, num_state_bins, num_action_bins):
        self.num_state_bins = num_state_bins
        self.num_action_bins = num_action_bins

    def precondition(self, current_state, action):
        return True

    def effect(self, current_state, action):
        bins = list(current_state.observed_bins())
        center = self.num_action_bins[0] // 2
        direction = 1 if action.bins[0] > center else -1
        next_bin = bins[0] + direction
        next_bin = max(0, min(self.num_state_bins[0] - 1, next_bin))
        bins[0] = DiscreteDistribution(support=[next_bin])
        current_state.bins = tuple(bins)

def build_laws(num_state_bins, num_action_bins):
    return [ActionMovesFirstStateBin(num_state_bins, num_action_bins)]
```
"""


def make_dataset() -> MuJoCoTransitions:
    return MuJoCoTransitions(
        states=np.array(
            [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.5, 0.0]],
            dtype=np.float32,
        ),
        actions=np.array([[-1.0], [1.0], [1.0], [-1.0]], dtype=np.float32),
        next_states=np.array(
            [[0.5, 0.0], [1.0, 0.0], [1.5, 0.0], [1.0, 0.0]],
            dtype=np.float32,
        ),
    )


def test_compile_onelife_mujoco_laws_and_evaluate():
    dataset = make_dataset()
    discretizer = MuJoCoDiscretizer.fit(dataset, state_bins=5, action_bins=3)
    laws = compile_onelife_mujoco_laws(
        extract_python_code(GENERATED_CODE),
        discretizer,
    )
    metrics = evaluate_onelife_llm_baseline(laws, discretizer, dataset)

    assert len(laws) == 1
    assert math.isfinite(metrics["mean_log_probability"])
    assert 0.0 <= metrics["bin_accuracy"] <= 1.0
    assert metrics["num_laws"] == 1.0


def test_compile_onelife_mujoco_laws_skips_invalid_laws():
    code = """
class BadOutOfRangeLaw(BinnedMuJoCoLaw):
    def effect(self, current_state, action):
        bins = list(current_state.observed_bins())
        bins[0] = DiscreteDistribution(support=[999])
        current_state.bins = tuple(bins)

class GoodIdentityLaw(BinnedMuJoCoLaw):
    def effect(self, current_state, action):
        bins = list(current_state.observed_bins())
        bins[0] = DiscreteDistribution(support=[bins[0]])
        current_state.bins = tuple(bins)

def build_laws(num_state_bins, num_action_bins):
    return [BadOutOfRangeLaw(), GoodIdentityLaw()]
"""
    dataset = make_dataset()
    discretizer = MuJoCoDiscretizer.fit(dataset, state_bins=5, action_bins=3)

    with pytest.warns(RuntimeWarning, match="Skipping invalid LLM-generated"):
        laws = compile_onelife_mujoco_laws(code, discretizer)

    assert [law.__name__ for law in laws] == ["GoodIdentityLaw"]


def test_compile_onelife_mujoco_laws_skips_non_law_objects():
    code = """
class GoodIdentityLaw(BinnedMuJoCoLaw):
    def effect(self, current_state, action):
        bins = list(current_state.observed_bins())
        bins[0] = DiscreteDistribution(support=[bins[0]])
        current_state.bins = tuple(bins)

def build_laws(num_state_bins, num_action_bins):
    return [GoodIdentityLaw, GoodIdentityLaw()]
"""
    dataset = make_dataset()
    discretizer = MuJoCoDiscretizer.fit(dataset, state_bins=5, action_bins=3)

    with pytest.warns(RuntimeWarning, match="expected BinnedMuJoCoLaw instance"):
        laws = compile_onelife_mujoco_laws(code, discretizer)

    assert [law.__name__ for law in laws] == ["GoodIdentityLaw"]


def test_onelife_llm_synthesizer_calls_mock_client():
    requests = []

    def fake_client(request):
        requests.append(request)
        return SimpleNamespace(
            choices=[SimpleNamespace(message={"content": GENERATED_CODE})]
        )

    dataset = make_dataset()
    discretizer = MuJoCoDiscretizer.fit(dataset, state_bins=5, action_bins=3)
    bundle = LLMOneLifeMuJoCoSynthesizer(
        llm_params=OpenAILiteLlmParams(api_key="test", model_slug="gpt-4o-mini"),
        llm_client=fake_client,
    ).synthesize_from_dataset(
        dataset,
        discretizer,
        LLMOneLifeSynthesisConfig(env_id="Fake-v0", sample_count=2),
    )

    assert len(requests) == 1
    assert "Fake-v0" in bundle.prompt
    assert len(bundle.laws) == 1
