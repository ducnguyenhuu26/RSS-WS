from __future__ import annotations

import torch

from onelife.program_residual import (
    IslandSearchConfig,
    KinematicPositionLaw,
    LLMLawSynthesisConfig,
    LLMSynthesizedLaws,
    TransitionBatch,
    synthesize_with_island_search,
)


class FakeSynthesizer:
    def __init__(self) -> None:
        self.calls = 0
        self.prompts: list[str] = []

    def synthesize_from_batch(self, batch, config):
        self.calls += 1
        self.prompts.append(config.extra_instructions)
        return LLMSynthesizedLaws(
            laws=(
                KinematicPositionLaw(
                    position_indices=[0],
                    velocity_indices=[1],
                    dt=config.dt,
                    confidence=1.0,
                ),
            ),
            code=f"# {config.niche}",
            prompt=config.extra_instructions,
            raw_response="",
        )


def test_island_search_returns_best_valid_symbolic_bundle():
    states = torch.tensor([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]], dtype=torch.float32)
    actions = torch.zeros(3, 1)
    next_states = torch.tensor([[0.1, 1.0], [1.2, 2.0], [2.3, 3.0]], dtype=torch.float32)
    batch = TransitionBatch(states=states, actions=actions, next_states=next_states)
    synthesizer = FakeSynthesizer()

    result = synthesize_with_island_search(
        synthesizer=synthesizer,  # type: ignore[arg-type]
        batch=batch,
        config=LLMLawSynthesisConfig(
            env_id="Fake-v0",
            dt=0.1,
            sample_count=3,
            validation_sample_count=3,
            max_validation_mse_ratio=10.0,
        ),
        search_config=IslandSearchConfig(
            candidates_per_niche=1,
            generations=1,
            island_size=2,
            validation_sample_count=3,
        ),
    )

    assert synthesizer.calls == 4
    assert result.bundle.laws
    assert result.summary["num_candidates"] >= 4
    assert "best_niche" in result.summary
    assert all("law_name string only" in prompt for prompt in synthesizer.prompts)
    assert all("state[i] and action[k] integer indexing only" in prompt for prompt in synthesizer.prompts)
