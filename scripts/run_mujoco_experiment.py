from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from statistics import mean

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

from onelife.litellm_utils import GeminiLiteLlmParams, OpenAILiteLlmParams
from onelife.mujoco_symbolic_adapter import (
    MuJoCoDiscretizer,
    make_onelife_mujoco_law_mixture,
    make_poe_mujoco_baseline,
    to_law_symbolic_transitions,
    to_poe_symbolic_transitions,
)
from onelife.program_residual import (
    LLMLawSynthesisConfig,
    LLMSymbolicLawSynthesizer,
    MuJoCoCollectionConfig,
    ProgramResidualTrainerConfig,
    ProgramResidualWorldModel,
    ResidualMLP,
    SymbolicProgram,
    TransitionBatch,
    collect_mujoco_dataset,
    fit_supervised,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a small MuJoCo program-residual experiment."
    )
    parser.add_argument("--env-id", default="Hopper-v5")
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--test-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--residual-l2-weight", type=float, default=1e-3)
    parser.add_argument("--hidden-sizes", default="128,128")
    parser.add_argument(
        "--symbolic-source",
        choices=["empty", "llm"],
        default="empty",
        help="empty = no symbolic law; llm = synthesize laws before training.",
    )
    parser.add_argument("--llm-provider", choices=["gemini", "openai"], default="gemini")
    parser.add_argument("--llm-model-slug", default=None)
    parser.add_argument("--llm-sample-count", type=int, default=8)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--state-bins", type=int, default=21)
    parser.add_argument("--action-bins", type=int, default=11)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/mujoco_experiment.json"),
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not args.verbose:
        logger.disable("onelife")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_dataset = collect_mujoco_dataset(
        config=MuJoCoCollectionConfig(
            env_id=args.env_id,
            num_steps=args.train_steps,
            seed=args.seed,
        )
    )
    test_dataset = collect_mujoco_dataset(
        config=MuJoCoCollectionConfig(
            env_id=args.env_id,
            num_steps=args.test_steps,
            seed=args.seed + 1,
        )
    )

    program, llm_code = build_symbolic_program(
        args=args,
        train_dataset=train_dataset,
    )
    residual = ResidualMLP(
        state_dim=train_dataset.state_dim,
        action_dim=train_dataset.action_dim,
        hidden_sizes=parse_hidden_sizes(args.hidden_sizes),
    )
    model = ProgramResidualWorldModel(
        state_dim=train_dataset.state_dim,
        action_dim=train_dataset.action_dim,
        program=program,
        residual_model=residual,
    )
    batches = make_batches(
        train_dataset.iter_torch_batches(
            batch_size=args.batch_size,
            shuffle=True,
            seed=args.seed,
        )
    )
    history = fit_supervised(
        model=model,
        batches=batches,
        config=ProgramResidualTrainerConfig(
            learning_rate=args.learning_rate,
            residual_l2_weight=args.residual_l2_weight,
        ),
        num_epochs=args.epochs,
    )

    continuous_metrics = evaluate_program_residual(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        state_bins=args.state_bins,
        action_bins=args.action_bins,
    )
    symbolic_metrics = evaluate_symbolic_baselines(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        state_bins=args.state_bins,
        action_bins=args.action_bins,
    )
    results = {
        "env_id": args.env_id,
        "seed": args.seed,
        "train_steps": args.train_steps,
        "test_steps": args.test_steps,
        "symbolic_source": args.symbolic_source,
        "state_dim": train_dataset.state_dim,
        "action_dim": train_dataset.action_dim,
        "final_train_loss": history[-1].loss if history else None,
        "program_residual": continuous_metrics,
        "symbolic_baselines": symbolic_metrics,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if llm_code is not None:
        law_path = args.output.with_suffix(".laws.py")
        law_path.write_text(llm_code, encoding="utf-8")
        results["llm_law_path"] = str(law_path)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(json.dumps(results, indent=2))


def build_symbolic_program(
    args: argparse.Namespace,
    train_dataset,
) -> tuple[SymbolicProgram, str | None]:
    if args.symbolic_source == "empty":
        return SymbolicProgram(state_dim=train_dataset.state_dim, laws=[]), None

    if args.llm_provider == "openai":
        llm_params = OpenAILiteLlmParams(
            model_slug=args.llm_model_slug or "gpt-4.1-mini",
        )
    else:
        llm_params = GeminiLiteLlmParams(
            model_slug=args.llm_model_slug or "gemini-2.5-flash",
        )
    bundle = LLMSymbolicLawSynthesizer(llm_params=llm_params).synthesize_from_mujoco_dataset(
        train_dataset,
        LLMLawSynthesisConfig(
            env_id=args.env_id,
            dt=args.dt,
            sample_count=args.llm_sample_count,
        ),
    )
    return bundle.build_program(state_dim=train_dataset.state_dim), bundle.code


def evaluate_program_residual(
    model: ProgramResidualWorldModel,
    train_dataset,
    test_dataset,
    state_bins: int,
    action_bins: int,
) -> dict[str, float]:
    states, actions, next_states = test_dataset.to_torch()
    model.eval()
    with torch.no_grad():
        output = model(states, actions)
        identity_mse = F.mse_loss(states, next_states)
        program_mse = F.mse_loss(output.program_next_state, next_states)
        prediction_mse = F.mse_loss(output.prediction, next_states)
    discretizer = MuJoCoDiscretizer.fit(
        train_dataset,
        state_bins=state_bins,
        action_bins=action_bins,
    )
    return {
        "identity_mse": float(identity_mse.cpu()),
        "program_only_mse": float(program_mse.cpu()),
        "program_residual_mse": float(prediction_mse.cpu()),
        "mean_unknown_fraction": float(output.unknown_mask.float().mean().cpu()),
        "bin_accuracy": symbolic_bin_accuracy(
            (
                discretizer.digitize_state(prediction).observed_bins()
                for prediction in output.prediction.detach().cpu().numpy()
            ),
            (
                discretizer.digitize_state(next_state).observed_bins()
                for next_state in test_dataset.next_states
            ),
        ),
    }


def evaluate_symbolic_baselines(
    train_dataset,
    test_dataset,
    state_bins: int,
    action_bins: int,
) -> dict[str, dict[str, float]]:
    discretizer = MuJoCoDiscretizer.fit(
        train_dataset,
        state_bins=state_bins,
        action_bins=action_bins,
    )
    onelife_model = make_onelife_mujoco_law_mixture(discretizer)
    poe_model = make_poe_mujoco_baseline(discretizer)
    law_transitions = to_law_symbolic_transitions(test_dataset, discretizer)
    poe_transitions = to_poe_symbolic_transitions(test_dataset, discretizer)
    return {
        "onelife_law_mixture": {
            "mean_log_probability": mean(
                onelife_model.evaluate_log_probability(
                    transition.prev_state,
                    transition.action,
                    transition.next_state,
                )
                for transition in law_transitions
            ),
            "bin_accuracy": symbolic_bin_accuracy(
                (
                    onelife_model.sample_next_state(t.prev_state, t.action).observed_bins()
                    for t in law_transitions
                ),
                (t.next_state.observed_bins() for t in law_transitions),
            ),
        },
        "poe_world": {
            "mean_log_probability": mean(
                poe_model.evaluate_log_probability(
                    transition.prev_metadata,
                    transition.action,
                    transition.next_metadata,
                )
                for transition in poe_transitions
            ),
            "bin_accuracy": symbolic_bin_accuracy(
                (
                    poe_model.sample_next_state(
                        t.prev_metadata,
                        t.action,
                    ).observed_bins()
                    for t in poe_transitions
                ),
                (t.next_metadata.observed_bins() for t in poe_transitions),
            ),
        },
    }


def symbolic_bin_accuracy(
    predicted: Iterable[tuple[int, ...]],
    target: Iterable[tuple[int, ...]],
) -> float:
    total = 0
    correct = 0
    for pred_bins, target_bins in zip(predicted, target):
        for pred_bin, target_bin in zip(pred_bins, target_bins):
            total += 1
            correct += int(pred_bin == target_bin)
    return correct / total if total else 0.0


def make_batches(
    raw_batches: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> list[TransitionBatch]:
    return [
        TransitionBatch(states=states, actions=actions, next_states=next_states)
        for states, actions, next_states in raw_batches
    ]


def parse_hidden_sizes(raw: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not values:
        raise ValueError("hidden sizes must not be empty")
    return values


if __name__ == "__main__":
    main()
