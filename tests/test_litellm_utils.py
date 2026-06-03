from __future__ import annotations

from types import SimpleNamespace

from onelife.litellm_utils import (
    LLMCallTracker,
    LiteLlmMessage,
    LiteLlmParamsBase,
    LiteLlmRequest,
)


class FakeRequest(LiteLlmRequest):
    def __call__(self):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message={"content": "```python\ncode\n```"},
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=11,
                completion_tokens=7,
                total_tokens=18,
            ),
        )


def test_llm_call_tracker_counts_logical_calls_and_usage():
    tracker = LLMCallTracker()
    content = "```python\ncode\n```"
    request = FakeRequest(
        messages=[
            LiteLlmMessage(role="system", content="system"),
            LiteLlmMessage(role="user", content="prompt"),
        ],
        params=LiteLlmParamsBase(
            provider="test",
            model_slug="fake",
            api_key="test",
        ),
    )

    response = tracker(request)

    assert response.choices[0].message["content"].startswith("```python")
    assert tracker.as_dict() == {
        "calls": 1,
        "successful_calls": 1,
        "failed_calls": 0,
        "prompt_messages": 2,
        "prompt_chars": 12,
        "response_chars": len(content),
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
    }
