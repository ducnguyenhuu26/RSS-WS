from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cloudpickle
import numpy as np
import torch

from onelife.litellm_utils import (
    GeminiLiteLlmParams,
    LiteLlmMessage,
    LiteLlmParamsBase,
    LiteLlmRequest,
)
from onelife.local_code_execution import ExecWithLimitedNamespace
from onelife.mujoco_dataset import MuJoCoTransitions
from onelife.mujoco_symbolic_adapter import (
    BinnedMuJoCoAction,
    BinnedMuJoCoState,
    MuJoCoBinnedObservableExtractor,
    MuJoCoDiscretizer,
    to_law_symbolic_transitions,
)
from onelife.our_method.core import WeightedLaw
from onelife.our_method.world_modeling import LawMixture
from onelife.poe_world.core import DiscreteDistribution


@dataclass(frozen=True)
class LLMOneLifeSynthesisConfig:
    env_id: str
    sample_count: int = 8
    max_prompt_float_precision: int = 4
    max_tokens: int = 2500


@dataclass(frozen=True)
class LLMOneLifeLaws:
    laws: tuple["BinnedMuJoCoLaw", ...]
    code: str
    prompt: str
    raw_response: str

    def build_law_mixture(
        self,
        discretizer: MuJoCoDiscretizer,
        weight: float = 1.0,
    ) -> LawMixture[BinnedMuJoCoState, BinnedMuJoCoAction]:
        return LawMixture(
            observable_extractor=MuJoCoBinnedObservableExtractor.from_discretizer(
                discretizer
            ),
            weighted_laws=[
                WeightedLaw(law=law, weight=weight, is_fitted=True)
                for law in self.laws
            ],
        )


LLMClient = Callable[[LiteLlmRequest], Any]


class BinnedMuJoCoLaw:
    """Base class for LLM-generated OneLife-style laws on binned MuJoCo states."""

    def precondition(
        self,
        current_state: BinnedMuJoCoState,
        action: BinnedMuJoCoAction,
    ) -> bool:
        return True

    def effect(
        self,
        current_state: BinnedMuJoCoState,
        action: BinnedMuJoCoAction,
    ) -> None:
        raise NotImplementedError

    @property
    def __source_code__(self) -> str:
        return self.__class__.__name__

    @property
    def __name__(self) -> str:
        return self.__class__.__name__

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file:
            cloudpickle.dump(self, file)

    @classmethod
    def load(cls, path: str | Path) -> "BinnedMuJoCoLaw":
        with Path(path).open("rb") as file:
            instance = cloudpickle.load(file)
        if not isinstance(instance, cls):
            raise TypeError(
                f"File '{path}' contained {type(instance).__name__}, "
                f"not {cls.__name__}."
            )
        return instance


class LLMOneLifeMuJoCoSynthesizer:
    """Offline LLM proposer for OneLife-style precondition/effect laws."""

    def __init__(
        self,
        llm_params: LiteLlmParamsBase | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.llm_params = llm_params
        self.llm_client = llm_client or (lambda request: request())

    def synthesize_from_dataset(
        self,
        transitions: MuJoCoTransitions,
        discretizer: MuJoCoDiscretizer,
        config: LLMOneLifeSynthesisConfig,
    ) -> LLMOneLifeLaws:
        prompt = build_onelife_mujoco_prompt(
            transitions=transitions,
            discretizer=discretizer,
            config=config,
        )
        request = LiteLlmRequest(
            messages=[
                LiteLlmMessage(role="system", content=_SYSTEM_PROMPT),
                LiteLlmMessage(role="user", content=prompt),
            ],
            params=self._params_with_config(config),
        )
        response = self.llm_client(request)
        raw_response = _response_text(response)
        code = extract_python_code(raw_response)
        laws = compile_onelife_mujoco_laws(
            code=code,
            discretizer=discretizer,
        )
        return LLMOneLifeLaws(
            laws=tuple(laws),
            code=code,
            prompt=prompt,
            raw_response=raw_response,
        )

    def _params_with_config(
        self,
        config: LLMOneLifeSynthesisConfig,
    ) -> LiteLlmParamsBase:
        params = self.llm_params or GeminiLiteLlmParams()
        if params.max_tokens is None:
            params = params.model_copy(update={"max_tokens": config.max_tokens})
        return params


def build_onelife_mujoco_prompt(
    transitions: MuJoCoTransitions,
    discretizer: MuJoCoDiscretizer,
    config: LLMOneLifeSynthesisConfig,
) -> str:
    samples = _format_binned_samples(transitions, discretizer, config)
    return f"""
We are adapting OneLife to MuJoCo by discretizing continuous states/actions.
Your task is to propose OneLife-style symbolic laws over binned variables.

Environment: {config.env_id}
State dimensions: {discretizer.state_dim}
Action dimensions: {discretizer.action_dim}
State bin counts by dimension: {list(discretizer.num_state_bins)}
Action bin counts by dimension: {list(discretizer.num_action_bins)}

Return only Python code inside one ```python fenced block.
Do not import anything. The execution namespace already contains:
    BinnedMuJoCoLaw, BinnedMuJoCoState, BinnedMuJoCoAction, DiscreteDistribution

The code must define:

def build_laws(num_state_bins: tuple[int, ...], num_action_bins: tuple[int, ...]) -> list[BinnedMuJoCoLaw]:
    ...

Each law must inherit BinnedMuJoCoLaw and implement:
    precondition(self, current_state: BinnedMuJoCoState, action: BinnedMuJoCoAction) -> bool
    effect(self, current_state: BinnedMuJoCoState, action: BinnedMuJoCoAction) -> None

State/action access:
    current_state.bins is a tuple of integer bin ids.
    action.bins is a tuple of integer bin ids.

To predict a next-state attribute, replace that state bin with:
    DiscreteDistribution(support=[predicted_bin])

Leave dimensions unchanged if the law does not confidently model them.
Prefer compact interpretable rules. Do not predict every dimension unless the rule is meaningful.

Observed binned transitions:
{samples}
""".strip()


def compile_onelife_mujoco_laws(
    code: str,
    discretizer: MuJoCoDiscretizer,
) -> list[BinnedMuJoCoLaw]:
    executor = ExecWithLimitedNamespace(
        allowed_names={
            "BinnedMuJoCoLaw",
            "BinnedMuJoCoState",
            "BinnedMuJoCoAction",
            "DiscreteDistribution",
        },
        inherited_scope={
            "BinnedMuJoCoLaw": BinnedMuJoCoLaw,
            "BinnedMuJoCoState": BinnedMuJoCoState,
            "BinnedMuJoCoAction": BinnedMuJoCoAction,
            "DiscreteDistribution": DiscreteDistribution,
        },
    )
    executor(code)
    build_laws = executor.namespace.get("build_laws")
    if not callable(build_laws):
        raise ValueError("LLM code must define callable build_laws(...)")
    raw_laws = build_laws(
        tuple(discretizer.num_state_bins),
        tuple(discretizer.num_action_bins),
    )
    if not isinstance(raw_laws, list):
        raise TypeError("build_laws(...) must return list[BinnedMuJoCoLaw]")
    laws: list[BinnedMuJoCoLaw] = []
    for law in raw_laws:
        if not isinstance(law, BinnedMuJoCoLaw):
            raise TypeError(
                "build_laws(...) returned an object that is not a BinnedMuJoCoLaw: "
                f"{type(law).__name__}"
            )
        _smoke_test_law(law, discretizer)
        laws.append(law)
    return laws


def evaluate_onelife_llm_baseline(
    laws: tuple[BinnedMuJoCoLaw, ...] | list[BinnedMuJoCoLaw],
    discretizer: MuJoCoDiscretizer,
    test_dataset: MuJoCoTransitions,
) -> dict[str, float]:
    model = LLMOneLifeLaws(
        laws=tuple(laws),
        code="",
        prompt="",
        raw_response="",
    ).build_law_mixture(discretizer)
    transitions = to_law_symbolic_transitions(test_dataset, discretizer)
    log_probs = [
        model.evaluate_log_probability(
            transition.prev_state,
            transition.action,
            transition.next_state,
        )
        for transition in transitions
    ]
    total = 0
    correct = 0
    for transition in transitions:
        predicted = model.sample_next_state(
            transition.prev_state,
            transition.action,
        ).observed_bins()
        target = transition.next_state.observed_bins()
        for predicted_bin, target_bin in zip(predicted, target):
            total += 1
            correct += int(predicted_bin == target_bin)
    return {
        "mean_log_probability": float(np.mean(log_probs)) if log_probs else 0.0,
        "bin_accuracy": correct / total if total else 0.0,
        "num_laws": float(len(laws)),
    }


def extract_python_code(raw_response: str) -> str:
    fenced = re.search(r"```(?:python)?\s*(.*?)```", raw_response, flags=re.DOTALL)
    if fenced is not None:
        return fenced.group(1).strip()
    return raw_response.strip()


def _smoke_test_law(
    law: BinnedMuJoCoLaw,
    discretizer: MuJoCoDiscretizer,
) -> None:
    state = BinnedMuJoCoState(tuple(0 for _ in range(discretizer.state_dim)))
    action = BinnedMuJoCoAction(tuple(0 for _ in range(discretizer.action_dim)))
    if not law.precondition(state, action):
        return
    law.effect(state, action)
    if len(state.bins) != discretizer.state_dim:
        raise ValueError(f"{law.__name__} changed state dimensionality")
    for dim, value in enumerate(state.bins):
        if isinstance(value, DiscreteDistribution):
            support = value.support
            if np.any(support < 0) or np.any(support >= discretizer.num_state_bins[dim]):
                raise ValueError(f"{law.__name__} predicted out-of-range bin")
        elif not isinstance(value, int):
            raise TypeError(
                f"{law.__name__} left invalid state value type: {type(value).__name__}"
            )


def _format_binned_samples(
    transitions: MuJoCoTransitions,
    discretizer: MuJoCoDiscretizer,
    config: LLMOneLifeSynthesisConfig,
) -> str:
    num_samples = min(int(config.sample_count), transitions.num_steps)
    if num_samples <= 0:
        raise ValueError("cannot synthesize laws from an empty transition dataset")
    lines = []
    for idx in range(num_samples):
        prev_state = discretizer.digitize_state(transitions.states[idx]).observed_bins()
        action = discretizer.digitize_action(transitions.actions[idx]).bins
        next_state = discretizer.digitize_state(
            transitions.next_states[idx]
        ).observed_bins()
        changed_dims = [
            dim
            for dim, (prev_bin, next_bin) in enumerate(zip(prev_state, next_state))
            if prev_bin != next_bin
        ]
        lines.append(
            f"- sample {idx}: state_bins={prev_state}; action_bins={action}; "
            f"next_state_bins={next_state}; changed_dims={changed_dims}"
        )
    return "\n".join(lines)


def _response_text(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError("LLM response does not contain choices")
    choice = choices[0]
    message = getattr(choice, "message", None)
    if isinstance(message, dict):
        content = message.get("content")
    else:
        content = getattr(message, "content", None)
    if content is None:
        content = getattr(choice, "text", None)
    if not isinstance(content, str) or not content.strip():
        raise ValueError("LLM response does not contain text content")
    return content


_SYSTEM_PROMPT = """
You write concise, safe Python laws for a OneLife-style symbolic world model.
Return only law code. No prose. No imports. No file, network, shell, or eval calls.
Use the provided BinnedMuJoCoLaw and DiscreteDistribution APIs exactly.
""".strip()
