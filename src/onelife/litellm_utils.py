from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel, Field, ConfigDict
from typing import Literal, Optional, Any, cast
from litellm.types.utils import ModelResponse, Choices
import litellm
from ulid import ULID
import os


class LiteLlmMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


# This class exists to make the type checker happy.
# The .choices field in the ModelResponse type is a union
# of Choice and StreamingChoice. The options we give the model
# will in practice always result in a Choice, so we narrow the
# type here and avoid tedious assertions or isinstance checks
# elsewhere in the code.
class NonStreamingModelResponse(ModelResponse):
    choices: list[Choices]  # pyright: ignore[reportIncompatibleVariableOverride]


class LiteLlmParamsBase(BaseModel):
    """
    A class encapsulating most of the parameters we're interested in
    specifying for a LiteLLM model call.

    Usage:
    ```python
    params = LiteLlmParamsBase(
        provider="openai",
        model_slug="gpt-4o",
        api_key=os.environ["OPENAI_API_KEY"],
    )
    print(litellm.completion(
        model=params.model,
        messages=["Hello, world!"],
        **params.kwargs,
    ))
    ```
    """

    provider: str
    model_slug: str
    api_key: str
    n: int = 1
    stream: bool = False
    max_tokens: Optional[int] = None
    num_retries: int = 3
    api_base: Optional[str] = None

    @property
    def kwargs(self) -> dict[str, Any]:
        """
        Returns a dictionary of parameters that can be passed to
        litellm.completion.
        """
        # Exclude the model_slug and provider fields from the dump
        # because they have to be passed to litellm.completion after merging.
        _kwargs = self.model_dump(exclude={"model_slug", "provider"})
        return _kwargs

    @property
    def model(self) -> str:
        """
        Returns the model name as a string in the format LiteLLM expects.
        """
        return f"{self.provider}/{self.model_slug}"


class OpenAILiteLlmParams(LiteLlmParamsBase):
    provider: str = "openai"
    model_slug: str = "gpt-4.1-mini"
    api_key: str = Field(default_factory=lambda: os.environ["OPENAI_API_KEY"])


class GeminiLiteLlmParams(LiteLlmParamsBase):
    provider: str = "gemini"
    model_slug: str = "gemini-2.5-flash"
    api_key: str = Field(default_factory=lambda: os.environ["GEMINI_API_KEY"])


class HuggingFaceLiteLlmParams(LiteLlmParamsBase):
    provider: str = "huggingface"
    model_slug: str = "meta-llama/Llama-3.1-8B-Instruct"
    api_key: str = Field(default_factory=lambda: os.environ["HF_TOKEN"])


class VllmLiteLlmParams(LiteLlmParamsBase):
    provider: str = "hosted_vllm"
    model_slug: str = ""  # Replace with the name of the LoRA or model you want to use
    api_base: Optional[str] = "http://localhost:8000/v1"
    api_key: str = ""


class LiteLlmRequest(BaseModel):
    messages: list[LiteLlmMessage]
    params: LiteLlmParamsBase
    ulid: ULID = Field(default_factory=ULID)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __call__(self) -> NonStreamingModelResponse:
        response = litellm.completion(
            model=self.params.model,
            messages=[m.model_dump() for m in self.messages],
            **self.params.kwargs,
        )
        response = cast(ModelResponse, response)
        return NonStreamingModelResponse.model_validate(response, from_attributes=True)


@dataclass
class LLMCallTracker:
    """Tracks logical LiteLLM requests for experiment ablations."""

    calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    prompt_messages: int = 0
    prompt_chars: int = 0
    response_chars: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __call__(self, request: LiteLlmRequest) -> Any:
        self.calls += 1
        self.prompt_messages += len(request.messages)
        self.prompt_chars += sum(len(message.content) for message in request.messages)
        try:
            response = request()
        except Exception:
            self.failed_calls += 1
            raise
        self.successful_calls += 1
        self.response_chars += len(_response_text_or_empty(response))
        self._record_usage(response)
        return response

    def client(self) -> Callable[[LiteLlmRequest], Any]:
        return self

    def as_dict(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "prompt_messages": self.prompt_messages,
            "prompt_chars": self.prompt_chars,
            "response_chars": self.response_chars,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }

    def _record_usage(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        self.prompt_tokens += _usage_int(usage, "prompt_tokens", "input_tokens")
        self.completion_tokens += _usage_int(
            usage,
            "completion_tokens",
            "output_tokens",
        )
        self.total_tokens += _usage_int(usage, "total_tokens")


def zero_llm_usage() -> dict[str, int]:
    return LLMCallTracker().as_dict()


def _usage_int(usage: Any, *names: str) -> int:
    if usage is None:
        return 0
    for name in names:
        if isinstance(usage, dict):
            value = usage.get(name)
        else:
            value = getattr(usage, name, None)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0
    return 0


def _response_text_or_empty(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        return ""
    choice = choices[0]
    message = getattr(choice, "message", None)
    if isinstance(message, dict):
        content = message.get("content")
    else:
        content = getattr(message, "content", None)
    if content is None:
        content = getattr(choice, "text", None)
    return content if isinstance(content, str) else ""


if __name__ == "__main__":
    litellm._turn_on_debug()
    params = GeminiLiteLlmParams()
    request = LiteLlmRequest(
        messages=[
            LiteLlmMessage(role="user", content="What's 1+1?"),
        ],
        params=params,
    )
    print(request())
