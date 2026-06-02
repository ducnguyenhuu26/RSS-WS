from __future__ import annotations

import argparse
from pathlib import Path

from onelife.litellm_utils import GeminiLiteLlmParams, OpenAILiteLlmParams
from onelife.program_residual import (
    LLMLawSynthesisConfig,
    LLMSymbolicLawSynthesizer,
    MuJoCoCollectionConfig,
    collect_mujoco_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthesize program-residual symbolic laws for a MuJoCo env."
    )
    parser.add_argument("--env-id", default="Hopper-v5")
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument(
        "--provider",
        choices=["gemini", "openai"],
        default="gemini",
    )
    parser.add_argument("--model-slug", default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/mujoco_synthesized_laws.py"),
    )
    args = parser.parse_args()

    dataset = collect_mujoco_dataset(
        config=MuJoCoCollectionConfig(
            env_id=args.env_id,
            num_steps=args.num_steps,
            seed=args.seed,
        )
    )
    if args.provider == "openai":
        llm_params = OpenAILiteLlmParams(
            model_slug=args.model_slug or "gpt-4.1-mini",
        )
    else:
        llm_params = GeminiLiteLlmParams(
            model_slug=args.model_slug or "gemini-2.5-flash",
        )

    synthesizer = LLMSymbolicLawSynthesizer(llm_params=llm_params)
    bundle = synthesizer.synthesize_from_mujoco_dataset(
        dataset,
        LLMLawSynthesisConfig(
            env_id=args.env_id,
            dt=args.dt,
            sample_count=args.sample_count,
        ),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(bundle.code, encoding="utf-8")
    print(f"validated {len(bundle.laws)} laws")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
