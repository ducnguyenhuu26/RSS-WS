from collections import deque
from typing import List, Optional, Callable, Literal, Protocol
from .balrog_interfaces import (
    Message,
    Observation,
    PromptBuilderProtocol,
    LlmClientProtocol,
)
import re
import copy
from pydantic import BaseModel
from typing_extensions import Self
from omegaconf import DictConfig
from .balrog_interfaces import EnvironmentProtocol, Text, Experience, OnResetExperience
from .typing_utils import implements
from typing import TypeVar, Union
from .balrog_interfaces import LLMResponse
from icecream import ic
from loguru import logger


class HistoryPromptBuilderConfig(BaseModel):
    max_text_history: int = 16
    max_image_history: int = 1
    system_prompt: Optional[str] = None
    max_cot_history: int = 1


class HistoryPromptBuilder:
    """Builds a prompt with a history of observations, actions, and reasoning.

    Maintains a configurable history of text, images, and chain-of-thought reasoning to
    construct prompt messages for conversational agents.
    """

    def __init__(
        self,
        config: HistoryPromptBuilderConfig,
    ):
        self.max_text_history = config.max_text_history
        self.max_image_history = config.max_image_history
        self.max_history = max(config.max_text_history, config.max_image_history)
        self.system_prompt = config.system_prompt
        self._events = deque(
            maxlen=self.max_history * 2
        )  # Stores observations and actions
        self._last_short_term_obs = None  # To store the latest short-term observation
        self.previous_reasoning = None
        self.max_cot_history = config.max_cot_history

    def update_instruction_prompt(self, instruction: str):
        """Set the system-level instruction prompt."""
        self.system_prompt = instruction

    def update_observation(self, obs: Observation):
        """Add an observation to the prompt history, which can include text, an image, or both."""
        long_term_context = obs.text.long_term_context
        self._last_short_term_obs = obs.text.short_term_context
        text = long_term_context

        image = obs.image

        # Add observation to events
        self._events.append(
            {
                "type": "observation",
                "text": text,
                "image": image,
            }
        )

    def update_action(self, action: str):
        """Add an action to the prompt history, including reasoning if available."""
        self._events.append(
            {
                "type": "action",
                "action": action,
                "reasoning": self.previous_reasoning,
            }
        )

    def update_reasoning(self, reasoning: str):
        """Set the reasoning text to be included with subsequent actions."""
        self.previous_reasoning = reasoning

    def reset(self):
        """Clear the event history."""
        self._events.clear()

    def get_prompt(self, icl_episodes=False) -> List[Message]:
        """Generate a list of Message objects representing the prompt.

        Returns:
            List[Message]: Messages constructed from the event history.
        """
        messages = []

        if self.system_prompt and not icl_episodes:
            messages.append(Message(role="user", content=self.system_prompt))

        # Determine which text observations to include
        text_needed = self.max_text_history
        for event in reversed(self._events):
            if event["type"] == "observation":
                if text_needed > 0 and event.get("text") is not None:
                    event["include_text"] = True
                    text_needed -= 1
                else:
                    event["include_text"] = False

        # Determine which image observations to include
        images_needed = self.max_image_history
        for event in reversed(self._events):
            if event["type"] == "observation":
                if images_needed > 0 and event.get("image") is not None:
                    event["include_image"] = True
                    images_needed -= 1
                else:
                    event["include_image"] = False

        # determine the reasoning to include
        reasoning_needed = self.max_cot_history
        for event in reversed(self._events):
            if event["type"] == "action":
                if reasoning_needed > 0 and event.get("reasoning") is not None:
                    reasoning_needed -= 1
                else:
                    event["reasoning"] = None

        # Process events to create messages
        for idx, event in enumerate(self._events):
            if event["type"] == "observation":
                message_parts = []

                if idx == len(self._events) - 1:
                    message_parts.append("Current Observation:")
                    if self._last_short_term_obs:
                        message_parts.append(self._last_short_term_obs)
                else:
                    message_parts.append("Observation:")

                if event.get("include_text", False):
                    message_parts.append(event["text"])

                image = None
                if event.get("include_image", False):
                    image = event["image"]
                    message_parts.append("Image observation provided.")

                content = "\n".join(message_parts)
                message = Message(role="user", content=content, attachment=image)

                # Clean up temporary flags
                for flag in ["include_text", "include_image"]:
                    if flag in event:
                        del event[flag]
            elif event["type"] == "action":
                if event.get("reasoning") is not None:
                    content = "Previous plan:\n" + event["reasoning"]
                else:
                    content = event["action"]
                message = Message(role="assistant", content=content)
            messages.append(message)

        return messages

    @classmethod
    def as_factory(cls, config: HistoryPromptBuilderConfig) -> Callable[[], Self]:
        return lambda: cls(config)


class NaiveAgent:
    """An agent that generates actions based on observations without complex reasoning."""

    def __init__(
        self,
        client: LlmClientProtocol,
        prompt_builder: PromptBuilderProtocol,
    ):
        """Initialize the NaiveAgent with a client and prompt builder."""
        self.client = client
        self.prompt_builder = prompt_builder

    def act(self, obs: Observation, prev_action: str | None = None) -> LLMResponse:
        """Generate the next action based on the observation and previous action.

        Args:
            obs (dict): The current observation in the environment.
            prev_action (str, optional): The previous action taken.

        Returns:
            str: The selected action from the LLM response.
        """
        if prev_action:
            self.prompt_builder.update_action(prev_action)

        self.prompt_builder.update_observation(obs)

        messages = self.prompt_builder.get_prompt()

        naive_instruction = """
You always have to output one of the above actions at a time and no other text. You always have to output an action until the episode terminates.
        """.strip()

        if messages and messages[-1].role == "user":
            messages[-1].content += "\n\n" + naive_instruction

        response = self.client.generate(messages)

        final_answer = self._extract_final_answer(response)

        return final_answer

    def _extract_final_answer(self, answer: LLMResponse) -> LLMResponse:
        """Sanitize the final answer, keeping only alphabetic characters.

        Args:
            answer (LLMResponse): The response from the LLM.

        Returns:
            LLMResponse: The sanitized response.
        """

        ic(answer.completion)

        # def filter_letters(input_string):
        #     return re.sub(r"[^a-zA-Z\s:]", "", input_string)

        final_answer = copy.deepcopy(answer)
        # final_answer = final_answer._replace(
        #     completion=filter_letters(final_answer.completion)
        # )

        ic(final_answer.completion)

        return final_answer

    def reset(self):
        self.prompt_builder.reset()

    @classmethod
    def as_factory(
        cls,
        client_factory: Callable[[], LlmClientProtocol],
        prompt_builder_factory: Callable[[], PromptBuilderProtocol],
    ) -> Callable[[], Self]:
        return lambda: cls(client_factory(), prompt_builder_factory())


class ReasoningNaiveAgent:
    """An agent that generates actions with reasoning, preserving the reasoning in the prompt builder."""

    def __init__(
        self,
        client: LlmClientProtocol,
        prompt_builder: HistoryPromptBuilder,
    ):
        """Initialize the ReasoningNaiveAgent with a client and prompt builder."""
        self.client = client
        self.prompt_builder = prompt_builder

    def act(self, obs: Observation, prev_action: str | None = None) -> LLMResponse:
        """Generate the next action based on the observation and previous action.

        Args:
            obs (dict): The current observation in the environment.
            prev_action (str, optional): The previous action taken.

        Returns:
            LLMResponse: The response containing the action and reasoning.
        """
        if prev_action:
            self.prompt_builder.update_action(prev_action)

        self.prompt_builder.update_observation(obs)

        messages = self.prompt_builder.get_prompt()

        reasoning_instruction = """
First, think about what's the best course of action step by step.
Then, provide your response in the following XML format:

<reasoning>
Your step-by-step reasoning here
</reasoning>

<action>
YOUR_CHOSEN_ACTION
</action>

Replace YOUR_CHOSEN_ACTION with exactly one of the listed actions. Output no other text outside these XML tags.
        """.strip()

        if messages and messages[-1].role == "user":
            messages[-1].content += "\n\n" + reasoning_instruction

        response = self.client.generate(messages)

        final_answer = self._extract_final_answer(response)

        return final_answer

    def _extract_final_answer(self, answer: LLMResponse) -> LLMResponse:
        """Extract reasoning and action from XML-tagged response.

        Args:
            answer (LLMResponse): The response from the LLM.

        Returns:
            LLMResponse: The response with extracted reasoning and action.
        """
        final_answer = copy.deepcopy(answer)

        completion_text = answer.completion

        # Log the raw completion for debugging
        logger.debug(f"Raw LLM completion: {completion_text}")

        # Extract reasoning from <reasoning> tags
        reasoning_match = re.search(
            r"<reasoning>(.*?)</reasoning>", completion_text, re.DOTALL
        )
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

        # Extract action from <action> tags
        action_match = re.search(r"<action>(.*?)</action>", completion_text, re.DOTALL)
        if action_match:
            action = action_match.group(1).strip()
        else:
            # Fallback: try to extract any text that looks like an action
            # Remove common action prefixes and clean up
            action = re.sub(
                r"^(ACTION:\s*|action:\s*)", "", completion_text, flags=re.IGNORECASE
            )
            action = re.sub(r"[^a-zA-Z\s:]", "", action).strip()

        # Log extracted reasoning and action
        logger.info(f"Extracted reasoning: {reasoning}")
        logger.info(f"Extracted action: {action}")

        # Update the prompt builder with the reasoning for future use
        if reasoning:
            self.prompt_builder.update_reasoning(reasoning)

        # Set the reasoning in the response and clean action as completion
        final_answer = final_answer._replace(reasoning=reasoning, completion=action)

        # Log the final answer
        logger.info(
            f"Final answer - reasoning: {final_answer.reasoning}, action: {final_answer.completion}"
        )

        return final_answer

    def reset(self):
        """Reset the prompt builder."""
        self.prompt_builder.reset()

    @classmethod
    def as_factory(
        cls,
        client_factory: Callable[[], LlmClientProtocol],
        prompt_builder_factory: Callable[[], HistoryPromptBuilder],
    ) -> Callable[[], Self]:
        return lambda: cls(client_factory(), prompt_builder_factory())


class BalrogEnvWrapper(Protocol):
    def reset(self, **kwargs) -> tuple[dict, dict]: ...
    def step(self, action: str) -> tuple[dict, float, bool, bool, dict]: ...
    def get_instruction_prompt(self, instructions: str | None) -> str: ...
    def check_action_validity(self, action: str) -> str: ...
    @property
    def failed_candidates(self) -> list[str]: ...
    def get_stats(self) -> dict: ...


# class TypedBalrogEnvironmentAdapter:
#     def __init__(self, env: BalrogEnvWrapper):
#         self.env = env

#     def reset(self, **kwargs) -> OnResetExperience[dict]:
#         obs, info = self.env.reset(**kwargs)
#         short_term_context = obs["text"]["short_term_context"]
#         long_term_context = obs["text"]["long_term_context"]
#         image = obs.get("image", None)
#         return OnResetExperience(
#             Observation(
#                 text=Text(
#                     short_term_context=short_term_context,
#                     long_term_context=long_term_context,
#                 ),
#                 image=image,
#                 obs=obs,
#             ),
#             info,
#         )

#     def step(self, action: str) -> Experience[dict]:
#         raw_obs, reward, terminated, truncated, info = self.env.step(action)
#         short_term_context = raw_obs["text"]["short_term_context"]
#         long_term_context = raw_obs["text"]["long_term_context"]
#         image = raw_obs.get("image", None)
#         obs = Observation(
#             text=Text(
#                 short_term_context=short_term_context,
#                 long_term_context=long_term_context,
#             ),
#             image=image,
#             obs=raw_obs,
#         )
#         return Experience(
#             obs=obs,
#             action=action,
#             reward=float(reward),
#             done=terminated,
#             truncated=truncated,
#             info=info,
#         )

#     def get_instruction_prompt(self, instructions: str | None = None) -> str:
#         return self.env.get_instruction_prompt(instructions)

#     def check_action_validity(self, candidate_action: str) -> str:
#         return self.env.check_action_validity(candidate_action)

#     @property
#     def failed_candidates(self) -> list[str]:
#         return self.env.failed_candidates

#     def get_stats(self) -> dict:
#         return self.env.get_stats()


# implements(EnvironmentProtocol)(TypedBalrogEnvironmentAdapter)


class EnvironmentConfig(BaseModel):
    name: str
    task: str

    def to_balrog_format(self) -> DictConfig:
        # Balrog requires the config to be in the format:
        # config.envs.{name_of_env}_kwargs
        return DictConfig(
            {
                "envs": {
                    f"{self.name}_kwargs": self.model_dump(exclude={"name", "task"}),
                }
            }
        )

    def create_balrog_env(
        self, balrog_env_factory: Callable[[str, str, DictConfig], BalrogEnvWrapper]
    ) -> BalrogEnvWrapper:
        return balrog_env_factory(self.name, self.task, self.to_balrog_format())
