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
