from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from onelife.litellm_utils import GeminiLiteLlmParams
from onelife.local_code_execution import SecurityException
from onelife.program_residual import (
    LLMLawSynthesisConfig,
    LLMSymbolicLawSynthesizer,
    TransitionBatch,
    compile_synthesized_laws,
    extract_python_code,
)


GENERATED_CODE = """
```python
class FirstActionDeltaLaw(ContinuousLaw):
    def __init__(self, confidence):
        super().__init__()
        self.confidence_value = float(confidence)

    def predict(self, state, action):
        indices = torch.tensor([0], dtype=torch.long, device=state.device)
        values = state[indices] + action[0:1]
        confidence = torch.full_like(values, self.confidence_value)
        return LawPrediction(
            indices=indices,
            values=values,
            confidence=confidence,
            law_name=self.law_name,
            value_kind="next_state",
        )

def build_laws(state_dim, action_dim, dt, confidence):
    return [FirstActionDeltaLaw(confidence)]
```
"""


def make_batch() -> TransitionBatch:
    return TransitionBatch(
        states=torch.tensor([[0.0, 1.0], [1.0, 2.0]], dtype=torch.float32),
        actions=torch.tensor([[0.5], [1.0]], dtype=torch.float32),
        next_states=torch.tensor([[0.5, 1.0], [2.0, 2.0]], dtype=torch.float32),
    )


def test_extract_python_code_from_fenced_response():
    assert "FirstActionDeltaLaw" in extract_python_code(GENERATED_CODE)
    assert "```" not in extract_python_code(GENERATED_CODE)


def test_compile_synthesized_laws_returns_executable_continuous_laws():
    laws = compile_synthesized_laws(
        code=extract_python_code(GENERATED_CODE),
        state_dim=2,
        action_dim=1,
        dt=0.05,
    )

    prediction = laws[0].predict(
        torch.tensor([2.0, 3.0]),
        torch.tensor([0.25]),
    )

    assert len(laws) == 1
    assert torch.allclose(prediction.indices, torch.tensor([0]))
    assert torch.allclose(prediction.values, torch.tensor([2.25]))


def test_synthesizer_calls_llm_and_builds_symbolic_program():
    requests = []

    def fake_client(request):
        requests.append(request)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message={"content": GENERATED_CODE},
                )
            ]
        )

    synthesizer = LLMSymbolicLawSynthesizer(
        llm_params=GeminiLiteLlmParams(api_key="test", max_tokens=123),
        llm_client=fake_client,
    )
    bundle = synthesizer.synthesize_from_batch(
        make_batch(),
        LLMLawSynthesisConfig(env_id="FakeMuJoCo-v0", sample_count=2),
    )
    program = bundle.build_program(state_dim=2)
    output = program(
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.5]),
    )

    assert len(requests) == 1
    assert "FakeMuJoCo-v0" in bundle.prompt
    assert len(bundle.laws) == 1
    assert torch.allclose(output.next_state, torch.tensor([1.5, 0.0]))
    assert torch.allclose(output.unknown_mask, torch.tensor([0.0, 1.0]))


def test_compile_synthesized_laws_rejects_imports():
    with pytest.raises(SecurityException):
        compile_synthesized_laws(
            code="import os\n\ndef build_laws(state_dim, action_dim, dt, confidence):\n    return []",
            state_dim=2,
            action_dim=1,
            dt=0.05,
        )


def test_compile_synthesized_laws_skips_bundle_when_build_laws_raises():
    code = """
def build_laws(state_dim, action_dim, dt, confidence):
    return [KinematicPositionLaw()]
"""

    with pytest.warns(RuntimeWarning, match="build_laws"):
        laws = compile_synthesized_laws(
            code=code,
            state_dim=2,
            action_dim=1,
            dt=0.05,
        )

    assert laws == []


def test_compile_synthesized_laws_skips_non_law_objects():
    code = """
class GoodLaw(ContinuousLaw):
    def predict(self, state, action):
        indices = torch.tensor([1], dtype=torch.long)
        values = state[1:2]
        confidence = torch.ones_like(values)
        return LawPrediction(indices, values, confidence, self.law_name, value_kind="next_state")

def build_laws(state_dim, action_dim, dt, confidence):
    return [object(), GoodLaw()]
"""

    with pytest.warns(RuntimeWarning, match="not a ContinuousLaw"):
        laws = compile_synthesized_laws(
            code=code,
            state_dim=2,
            action_dim=1,
            dt=0.05,
        )

    assert [law.law_name for law in laws] == ["GoodLaw"]


def test_compile_synthesized_laws_skips_invalid_llm_laws():
    code = """
class BadBatchedIndexLaw(ContinuousLaw):
    def predict(self, state, action):
        indices = torch.tensor([0], dtype=torch.long)
        values = state[:, 0]
        confidence = torch.ones_like(values)
        return LawPrediction(indices, values, confidence, self.law_name, value_kind="next_state")

class GoodLaw(ContinuousLaw):
    def predict(self, state, action):
        indices = torch.tensor([1], dtype=torch.long)
        values = state[1:2] + 0.0 * action[0:1]
        confidence = torch.ones_like(values)
        return LawPrediction(indices, values, confidence, self.law_name, value_kind="next_state")

def build_laws(state_dim, action_dim, dt, confidence):
    return [BadBatchedIndexLaw(), GoodLaw()]
"""

    with pytest.warns(RuntimeWarning, match="Skipping invalid LLM-generated law"):
        laws = compile_synthesized_laws(
            code=code,
            state_dim=2,
            action_dim=1,
            dt=0.05,
        )

    assert [law.law_name for law in laws] == ["GoodLaw"]


def test_compile_synthesized_laws_skips_laws_that_fail_batch_validation():
    code = """
class BadLaw(ContinuousLaw):
    def predict(self, state, action):
        indices = torch.tensor([0], dtype=torch.long)
        values = state[0:1] + 100.0
        confidence = torch.ones_like(values)
        return LawPrediction(indices, values, confidence, self.law_name, value_kind="next_state")

class GoodLaw(ContinuousLaw):
    def predict(self, state, action):
        indices = torch.tensor([0], dtype=torch.long)
        values = state[0:1] + action[0:1]
        confidence = torch.ones_like(values)
        return LawPrediction(indices, values, confidence, self.law_name, value_kind="next_state")

def build_laws(state_dim, action_dim, dt, confidence):
    return [BadLaw(), GoodLaw()]
"""

    with pytest.warns(RuntimeWarning, match="failed validation"):
        laws = compile_synthesized_laws(
            code=code,
            state_dim=2,
            action_dim=1,
            dt=0.05,
            validation_batch=make_batch(),
        )

    assert [law.law_name for law in laws] == ["GoodLaw"]
