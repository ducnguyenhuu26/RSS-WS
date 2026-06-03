from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym
from loguru import logger

from onelife.litellm_utils import (
    GeminiLiteLlmParams,
    LLMCallTracker,
    OpenAILiteLlmParams,
    zero_llm_usage,
)
from onelife.mujoco_symbolic_adapter import (
    BinnedMuJoCoAction,
    BinnedMuJoCoState,
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


@dataclass(frozen=True)
class PlannerEvaluationConfig:
    enabled: bool = False
    num_episodes: int = 2
    max_episode_steps: int = 200
    horizon: int = 5
    random_candidates: int = 64
    cem_candidates: int = 64
    cem_elites: int = 8
    cem_iterations: int = 2
    action_policy_seed_offset: int = 40_000


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
        "--device",
        default="auto",
        help="Torch device for neural training: auto, cpu, cuda, or cuda:0.",
    )
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
    parser.add_argument("--rollout-horizons", default="1,5,10,25,50")
    parser.add_argument("--rollout-num-rollouts", type=int, default=20)
    parser.add_argument(
        "--rollout-action-policy",
        choices=["zero", "random"],
        default="zero",
    )
    parser.add_argument("--rollout-stop-on-done", action="store_true")
    parser.add_argument("--planner-enable", action="store_true")
    parser.add_argument("--planner-episodes", type=int, default=2)
    parser.add_argument("--planner-max-steps", type=int, default=200)
    parser.add_argument("--planner-horizon", type=int, default=5)
    parser.add_argument("--planner-random-candidates", type=int, default=64)
    parser.add_argument("--planner-cem-candidates", type=int, default=64)
    parser.add_argument("--planner-cem-elites", type=int, default=8)
    parser.add_argument("--planner-cem-iterations", type=int, default=2)
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
    device = resolve_torch_device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

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

    program, llm_code, llm_usage = build_symbolic_program(
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
    ).to(device)
    batches = [
        batch.to(device)
        for batch in make_batches(
            train_dataset.iter_torch_batches(
                batch_size=args.batch_size,
                shuffle=True,
                seed=args.seed,
            )
        )
    ]
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
        env_id=args.env_id,
        seed=args.seed + 10_000,
        state_bins=args.state_bins,
        action_bins=args.action_bins,
        rollout_horizons=parse_int_list(args.rollout_horizons),
        rollout_num_rollouts=args.rollout_num_rollouts,
        rollout_action_policy=args.rollout_action_policy,
        rollout_stop_on_done=args.rollout_stop_on_done,
        planner_config=PlannerEvaluationConfig(
            enabled=args.planner_enable,
            num_episodes=args.planner_episodes,
            max_episode_steps=args.planner_max_steps,
            horizon=args.planner_horizon,
            random_candidates=args.planner_random_candidates,
            cem_candidates=args.planner_cem_candidates,
            cem_elites=args.planner_cem_elites,
            cem_iterations=args.planner_cem_iterations,
        ),
    )
    symbolic_metrics = evaluate_symbolic_baselines(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        state_bins=args.state_bins,
        action_bins=args.action_bins,
        env_id=args.env_id,
        seed=args.seed + 20_000,
        rollout_horizons=parse_int_list(args.rollout_horizons),
        rollout_num_rollouts=args.rollout_num_rollouts,
        rollout_action_policy=args.rollout_action_policy,
        rollout_stop_on_done=args.rollout_stop_on_done,
    )
    results = {
        "env_id": args.env_id,
        "seed": args.seed,
        "train_steps": args.train_steps,
        "test_steps": args.test_steps,
        "symbolic_source": args.symbolic_source,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else None,
        "llm_calls": int(llm_usage["calls"]),
        "llm_usage": llm_usage,
        "state_dim": train_dataset.state_dim,
        "action_dim": train_dataset.action_dim,
        "final_train_loss": history[-1].loss if history else None,
        "score": score_payload_from_metrics(continuous_metrics),
        "reward": reward_payload_from_metrics(continuous_metrics),
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
) -> tuple[SymbolicProgram, str | None, dict[str, int]]:
    if args.symbolic_source == "empty":
        return (
            SymbolicProgram(state_dim=train_dataset.state_dim, laws=[]),
            None,
            zero_llm_usage(),
        )

    if args.llm_provider == "openai":
        llm_params = OpenAILiteLlmParams(
            model_slug=args.llm_model_slug or "gpt-4.1-mini",
        )
    else:
        llm_params = GeminiLiteLlmParams(
            model_slug=args.llm_model_slug or "gemini-2.5-flash",
        )
    llm_tracker = LLMCallTracker()
    bundle = LLMSymbolicLawSynthesizer(
        llm_params=llm_params,
        llm_client=llm_tracker.client(),
    ).synthesize_from_mujoco_dataset(
        train_dataset,
        LLMLawSynthesisConfig(
            env_id=args.env_id,
            dt=args.dt,
            sample_count=args.llm_sample_count,
        ),
    )
    return (
        bundle.build_program(state_dim=train_dataset.state_dim),
        bundle.code,
        llm_tracker.as_dict(),
    )


def evaluate_program_residual(
    model: ProgramResidualWorldModel,
    train_dataset,
    test_dataset,
    env_id: str,
    seed: int,
    state_bins: int,
    action_bins: int,
    rollout_horizons: tuple[int, ...] = (1, 5, 10, 25, 50),
    rollout_num_rollouts: int = 20,
    rollout_action_policy: str = "zero",
    rollout_stop_on_done: bool = False,
    planner_config: PlannerEvaluationConfig | None = None,
) -> dict[str, float]:
    states, actions, next_states = test_dataset.to_torch()
    device = module_device(model)
    states = states.to(device)
    actions = actions.to(device)
    next_states = next_states.to(device)
    model.eval()
    with torch.no_grad():
        output = model(states, actions)
        identity_mse = F.mse_loss(states, next_states)
        program_mse = F.mse_loss(output.program_next_state, next_states)
        prediction_mse = F.mse_loss(output.prediction, next_states)
    r2_metrics = one_step_r2_metrics(
        states=test_dataset.states,
        predictions=output.prediction.detach().cpu().numpy(),
        next_states=test_dataset.next_states,
    )
    discretizer = MuJoCoDiscretizer.fit(
        train_dataset,
        state_bins=state_bins,
        action_bins=action_bins,
    )
    metrics = {
        "identity_mse": float(identity_mse.cpu()),
        "program_only_mse": float(program_mse.cpu()),
        "program_residual_mse": float(prediction_mse.cpu()),
        "mean_unknown_fraction": float(output.unknown_mask.float().mean().cpu()),
        **r2_metrics,
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
    rollout_metrics = evaluate_open_loop_rollouts(
        model=model,
        env_id=env_id,
        seed=seed,
        discretizer=discretizer,
        horizons=rollout_horizons,
        num_rollouts=rollout_num_rollouts,
        action_policy=rollout_action_policy,
        stop_on_done=rollout_stop_on_done,
    )
    metrics.update(rollout_metrics)
    if planner_config is not None and planner_config.enabled:
        metrics.update(
            evaluate_continuous_planner_rewards(
                model=model,
                env_id=env_id,
                seed=seed + planner_config.action_policy_seed_offset,
                config=planner_config,
            )
        )
    return metrics


def evaluate_open_loop_rollouts(
    model: ProgramResidualWorldModel,
    env_id: str,
    seed: int,
    discretizer: MuJoCoDiscretizer,
    horizons: tuple[int, ...],
    num_rollouts: int,
    action_policy: str = "zero",
    stop_on_done: bool = False,
) -> dict[str, float]:
    if not horizons:
        return {}
    if num_rollouts <= 0:
        raise ValueError("rollout_num_rollouts must be positive")
    max_horizon = max(horizons)
    if max_horizon <= 0:
        raise ValueError("rollout horizons must be positive")

    sums = {horizon: 0.0 for horizon in horizons}
    counts = {horizon: 0 for horizon in horizons}
    bin_correct = {horizon: 0 for horizon in horizons}
    bin_total = {horizon: 0 for horizon in horizons}
    env = gym.make(env_id)
    try:
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(seed)
        model.eval()
        with torch.no_grad():
            for rollout_idx in range(num_rollouts):
                reset_result = env.reset(seed=seed + rollout_idx)
                observation = (
                    reset_result[0] if isinstance(reset_result, tuple) else reset_result
                )
                predicted_state = torch.as_tensor(
                    np.asarray(observation, dtype=np.float32).reshape(-1),
                    dtype=torch.float32,
                )
                for step in range(1, max_horizon + 1):
                    action = sample_rollout_action(env, action_policy)
                    step_result = env.step(action)
                    if len(step_result) == 5:
                        next_observation, _reward, terminated, truncated, _info = (
                            step_result
                        )
                        done = bool(terminated or truncated)
                    else:
                        next_observation, _reward, done, _info = step_result
                        done = bool(done)
                    action_tensor = torch.as_tensor(
                        action.reshape(-1),
                        dtype=torch.float32,
                    )
                    predicted_state = model.predict_next_state(
                        predicted_state,
                        action_tensor,
                    )
                    true_next_state = torch.as_tensor(
                        np.asarray(next_observation, dtype=np.float32).reshape(-1),
                        dtype=torch.float32,
                    )
                    if step in sums:
                        sums[step] += float(
                            F.mse_loss(predicted_state, true_next_state).cpu()
                        )
                        counts[step] += 1
                        predicted_bins = discretizer.digitize_state(
                            predicted_state.detach().cpu().numpy()
                        ).observed_bins()
                        target_bins = discretizer.digitize_state(
                            true_next_state.detach().cpu().numpy()
                        ).observed_bins()
                        for predicted_bin, target_bin in zip(
                            predicted_bins,
                            target_bins,
                        ):
                            bin_total[step] += 1
                            bin_correct[step] += int(predicted_bin == target_bin)
                    if done and stop_on_done:
                        break
    finally:
        env.close()
    metrics = {
        f"open_loop_mse_h{horizon}": sums[horizon] / counts[horizon]
        for horizon in horizons
        if counts[horizon] > 0
    }
    metrics.update(
        {
            f"open_loop_bin_accuracy_h{horizon}": bin_correct[horizon]
            / bin_total[horizon]
            for horizon in horizons
            if bin_total[horizon] > 0
        }
    )
    return metrics


def module_device(module: torch.nn.Module) -> torch.device:
    parameter = next(module.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(module.buffers(), None)
    if buffer is not None:
        return buffer.device
    return torch.device("cpu")


def resolve_torch_device(raw_device: str | torch.device | None) -> torch.device:
    raw = "auto" if raw_device is None else str(raw_device).strip().lower()
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(raw)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def evaluate_symbolic_baselines(
    train_dataset,
    test_dataset,
    state_bins: int,
    action_bins: int,
    env_id: str | None = None,
    seed: int = 0,
    rollout_horizons: tuple[int, ...] = (1, 5, 10, 25, 50),
    rollout_num_rollouts: int = 20,
    rollout_action_policy: str = "zero",
    rollout_stop_on_done: bool = False,
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
    metrics = {
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
    if env_id is not None:
        metrics["onelife_law_mixture"].update(
            evaluate_symbolic_open_loop_rollouts(
                model=onelife_model,
                discretizer=discretizer,
                env_id=env_id,
                seed=seed,
                horizons=rollout_horizons,
                num_rollouts=rollout_num_rollouts,
                action_policy=rollout_action_policy,
                stop_on_done=rollout_stop_on_done,
            )
        )
        metrics["poe_world"].update(
            evaluate_symbolic_open_loop_rollouts(
                model=poe_model,
                discretizer=discretizer,
                env_id=env_id,
                seed=seed,
                horizons=rollout_horizons,
                num_rollouts=rollout_num_rollouts,
                action_policy=rollout_action_policy,
                stop_on_done=rollout_stop_on_done,
            )
        )
    return metrics


def evaluate_symbolic_open_loop_rollouts(
    model: Any,
    discretizer: MuJoCoDiscretizer,
    env_id: str,
    seed: int,
    horizons: tuple[int, ...],
    num_rollouts: int,
    action_policy: str = "zero",
    stop_on_done: bool = False,
) -> dict[str, float]:
    if not horizons:
        return {}
    if num_rollouts <= 0:
        raise ValueError("rollout_num_rollouts must be positive")
    max_horizon = max(horizons)
    if max_horizon <= 0:
        raise ValueError("rollout horizons must be positive")

    correct = {horizon: 0 for horizon in horizons}
    total = {horizon: 0 for horizon in horizons}
    log_prob_sums = {horizon: 0.0 for horizon in horizons}
    log_prob_counts = {horizon: 0 for horizon in horizons}
    env = gym.make(env_id)
    try:
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(seed)
        for rollout_idx in range(num_rollouts):
            reset_result = env.reset(seed=seed + rollout_idx)
            observation = (
                reset_result[0] if isinstance(reset_result, tuple) else reset_result
            )
            predicted_state = discretizer.digitize_state(observation)
            cumulative_log_prob = 0.0
            for step in range(1, max_horizon + 1):
                action = sample_rollout_action(env, action_policy)
                action_bins = discretizer.digitize_action(action)
                step_result = env.step(action)
                if len(step_result) == 5:
                    next_observation, _reward, terminated, truncated, _info = step_result
                    done = bool(terminated or truncated)
                else:
                    next_observation, _reward, done, _info = step_result
                    done = bool(done)
                target_state = discretizer.digitize_state(next_observation)
                cumulative_log_prob += float(
                    model.evaluate_log_probability(
                        predicted_state,
                        action_bins,
                        target_state,
                    )
                )
                predicted_state = model.sample_next_state(
                    predicted_state,
                    action_bins,
                )
                if step in correct:
                    predicted_bins = predicted_state.observed_bins()
                    target_bins = target_state.observed_bins()
                    for predicted_bin, target_bin in zip(predicted_bins, target_bins):
                        total[step] += 1
                        correct[step] += int(predicted_bin == target_bin)
                    log_prob_sums[step] += cumulative_log_prob
                    log_prob_counts[step] += 1
                if done and stop_on_done:
                    break
    finally:
        env.close()
    metrics = {
        f"open_loop_bin_accuracy_h{horizon}": correct[horizon] / total[horizon]
        for horizon in horizons
        if total[horizon] > 0
    }
    metrics.update(
        {
            f"open_loop_mean_log_probability_h{horizon}": log_prob_sums[horizon]
            / log_prob_counts[horizon]
            for horizon in horizons
            if log_prob_counts[horizon] > 0
        }
    )
    return metrics


def one_step_r2_metrics(
    states: np.ndarray,
    predictions: np.ndarray,
    next_states: np.ndarray,
) -> dict[str, float]:
    true_delta = np.asarray(next_states, dtype=np.float64) - np.asarray(
        states, dtype=np.float64
    )
    pred_delta = np.asarray(predictions, dtype=np.float64) - np.asarray(
        states, dtype=np.float64
    )
    predictions = np.asarray(predictions, dtype=np.float64)
    next_states = np.asarray(next_states, dtype=np.float64)
    return {
        "one_step_delta_r2_uniform": r2_uniform(true_delta, pred_delta),
        "one_step_delta_r2_global": r2_global(true_delta, pred_delta),
        "one_step_next_state_r2_uniform": r2_uniform(next_states, predictions),
        "one_step_next_state_r2_global": r2_global(next_states, predictions),
    }


def r2_uniform(
    targets: np.ndarray,
    predictions: np.ndarray,
    eps: float = 1e-12,
) -> float:
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    if targets.shape != predictions.shape:
        raise ValueError("targets and predictions must have matching shape")
    if targets.ndim != 2:
        raise ValueError("R2 inputs must have shape [num_samples, num_dimensions]")

    residual_ss = np.sum(np.square(targets - predictions), axis=0)
    centered = targets - np.mean(targets, axis=0, keepdims=True)
    total_ss = np.sum(np.square(centered), axis=0)
    per_dim = np.where(
        total_ss > eps,
        1.0 - residual_ss / np.maximum(total_ss, eps),
        np.where(residual_ss <= eps, 1.0, 0.0),
    )
    return float(np.mean(per_dim))


def r2_global(
    targets: np.ndarray,
    predictions: np.ndarray,
    eps: float = 1e-12,
) -> float:
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    if targets.shape != predictions.shape:
        raise ValueError("targets and predictions must have matching shape")
    residual_ss = float(np.sum(np.square(targets - predictions)))
    centered = targets - float(np.mean(targets))
    total_ss = float(np.sum(np.square(centered)))
    if total_ss <= eps:
        return 1.0 if residual_ss <= eps else 0.0
    return float(1.0 - residual_ss / total_ss)


def evaluate_binned_model_r2(
    model: Any,
    discretizer: MuJoCoDiscretizer,
    test_dataset,
) -> dict[str, float]:
    predictions: list[np.ndarray] = []
    for state, action in zip(test_dataset.states, test_dataset.actions):
        binned_state = discretizer.digitize_state(state)
        binned_action = discretizer.digitize_action(action)
        predicted_binned = model.sample_next_state(binned_state, binned_action)
        predictions.append(discretizer.undigitize_state(predicted_binned))
    return one_step_r2_metrics(
        states=test_dataset.states,
        predictions=np.stack(predictions).astype(np.float32),
        next_states=test_dataset.next_states,
    )


def score_payload_from_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {
        "one_step_delta_r2_uniform": metrics["one_step_delta_r2_uniform"],
        "one_step_delta_r2_global": metrics["one_step_delta_r2_global"],
        "one_step_next_state_r2_uniform": metrics["one_step_next_state_r2_uniform"],
        "one_step_next_state_r2_global": metrics["one_step_next_state_r2_global"],
    }


def reward_payload_from_metrics(metrics: dict[str, float]) -> dict[str, float]:
    reward_keys = [
        "random_mpc_return_mean",
        "random_mpc_return_std",
        "cem_mpc_return_mean",
        "cem_mpc_return_std",
        "random_mpc_episode_length_mean",
        "cem_mpc_episode_length_mean",
    ]
    return {key: metrics[key] for key in reward_keys if key in metrics}


def evaluate_continuous_planner_rewards(
    model: ProgramResidualWorldModel,
    env_id: str,
    seed: int,
    config: PlannerEvaluationConfig,
) -> dict[str, float]:
    def predict_next(planner_state: Any, action: np.ndarray) -> np.ndarray:
        prediction = model.predict_next_state(
            np.asarray(planner_state, dtype=np.float32),
            np.asarray(action, dtype=np.float32),
        )
        return prediction.detach().cpu().numpy().astype(np.float32)

    def observation(planner_state: Any) -> np.ndarray:
        return np.asarray(planner_state, dtype=np.float32)

    return evaluate_planner_rewards(
        env_id=env_id,
        seed=seed,
        config=config,
        initial_planner_state=lambda obs: np.asarray(obs, dtype=np.float32).reshape(-1),
        predict_next=predict_next,
        observation_for_reward=observation,
    )


def evaluate_binned_planner_rewards(
    model: Any,
    discretizer: MuJoCoDiscretizer,
    env_id: str,
    seed: int,
    config: PlannerEvaluationConfig,
) -> dict[str, float]:
    def predict_next(planner_state: Any, action: np.ndarray) -> BinnedMuJoCoState:
        binned_state = (
            planner_state
            if isinstance(planner_state, BinnedMuJoCoState)
            else discretizer.digitize_state(planner_state)
        )
        binned_action: BinnedMuJoCoAction = discretizer.digitize_action(action)
        return model.sample_next_state(binned_state, binned_action)

    def observation(planner_state: Any) -> np.ndarray:
        if isinstance(planner_state, BinnedMuJoCoState):
            return discretizer.undigitize_state(planner_state)
        return np.asarray(planner_state, dtype=np.float32).reshape(-1)

    return evaluate_planner_rewards(
        env_id=env_id,
        seed=seed,
        config=config,
        initial_planner_state=lambda obs: discretizer.digitize_state(obs),
        predict_next=predict_next,
        observation_for_reward=observation,
    )


def evaluate_planner_rewards(
    env_id: str,
    seed: int,
    config: PlannerEvaluationConfig,
    initial_planner_state: Callable[[np.ndarray], Any],
    predict_next: Callable[[Any, np.ndarray], Any],
    observation_for_reward: Callable[[Any], np.ndarray],
) -> dict[str, float]:
    random_returns, random_lengths = _run_planner_episodes(
        planner_name="random_mpc",
        env_id=env_id,
        seed=seed,
        config=config,
        initial_planner_state=initial_planner_state,
        predict_next=predict_next,
        observation_for_reward=observation_for_reward,
    )
    cem_returns, cem_lengths = _run_planner_episodes(
        planner_name="cem_mpc",
        env_id=env_id,
        seed=seed + 10_000,
        config=config,
        initial_planner_state=initial_planner_state,
        predict_next=predict_next,
        observation_for_reward=observation_for_reward,
    )
    return {
        "random_mpc_return_mean": _mean(random_returns),
        "random_mpc_return_std": _std(random_returns),
        "random_mpc_episode_length_mean": _mean(random_lengths),
        "cem_mpc_return_mean": _mean(cem_returns),
        "cem_mpc_return_std": _std(cem_returns),
        "cem_mpc_episode_length_mean": _mean(cem_lengths),
    }


def _run_planner_episodes(
    planner_name: str,
    env_id: str,
    seed: int,
    config: PlannerEvaluationConfig,
    initial_planner_state: Callable[[np.ndarray], Any],
    predict_next: Callable[[Any, np.ndarray], Any],
    observation_for_reward: Callable[[Any], np.ndarray],
) -> tuple[list[float], list[float]]:
    returns: list[float] = []
    lengths: list[float] = []
    rng = np.random.default_rng(seed)
    env = gym.make(env_id)
    try:
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(seed)
        action_low, action_high = _action_bounds(env)
        for episode_idx in range(config.num_episodes):
            reset_result = env.reset(seed=seed + episode_idx)
            observation = (
                reset_result[0] if isinstance(reset_result, tuple) else reset_result
            )
            total_reward = 0.0
            episode_length = 0
            for _ in range(config.max_episode_steps):
                current_observation = np.asarray(observation, dtype=np.float32).reshape(
                    -1
                )
                planner_state = initial_planner_state(current_observation)
                if planner_name == "random_mpc":
                    action = random_shooting_action(
                        planner_state=planner_state,
                        env_id=env_id,
                        rng=rng,
                        config=config,
                        action_low=action_low,
                        action_high=action_high,
                        predict_next=predict_next,
                        observation_for_reward=observation_for_reward,
                    )
                elif planner_name == "cem_mpc":
                    action = cem_mpc_action(
                        planner_state=planner_state,
                        env_id=env_id,
                        rng=rng,
                        config=config,
                        action_low=action_low,
                        action_high=action_high,
                        predict_next=predict_next,
                        observation_for_reward=observation_for_reward,
                    )
                else:
                    raise ValueError(f"unknown planner: {planner_name}")
                step_result = env.step(action.astype(np.float32))
                if len(step_result) == 5:
                    observation, reward, terminated, truncated, _info = step_result
                    done = bool(terminated or truncated)
                else:
                    observation, reward, done, _info = step_result
                    done = bool(done)
                total_reward += float(reward)
                episode_length += 1
                if done:
                    break
            returns.append(total_reward)
            lengths.append(float(episode_length))
    finally:
        env.close()
    return returns, lengths


def random_shooting_action(
    planner_state: Any,
    env_id: str,
    rng: np.random.Generator,
    config: PlannerEvaluationConfig,
    action_low: np.ndarray,
    action_high: np.ndarray,
    predict_next: Callable[[Any, np.ndarray], Any],
    observation_for_reward: Callable[[Any], np.ndarray],
) -> np.ndarray:
    action_sequences = rng.uniform(
        low=action_low,
        high=action_high,
        size=(config.random_candidates, config.horizon, action_low.shape[0]),
    ).astype(np.float32)
    scores = score_action_sequences(
        planner_state,
        action_sequences,
        env_id,
        predict_next,
        observation_for_reward,
    )
    return action_sequences[int(np.argmax(scores)), 0]


def cem_mpc_action(
    planner_state: Any,
    env_id: str,
    rng: np.random.Generator,
    config: PlannerEvaluationConfig,
    action_low: np.ndarray,
    action_high: np.ndarray,
    predict_next: Callable[[Any, np.ndarray], Any],
    observation_for_reward: Callable[[Any], np.ndarray],
) -> np.ndarray:
    action_dim = int(action_low.shape[0])
    mean_sequence = np.zeros((config.horizon, action_dim), dtype=np.float32)
    std_sequence = np.maximum((action_high - action_low) / 2.0, 1e-3)
    std_sequence = np.broadcast_to(std_sequence, mean_sequence.shape).copy()
    best_sequence = mean_sequence.copy()
    best_score = -float("inf")

    for _ in range(config.cem_iterations):
        samples = rng.normal(
            loc=mean_sequence,
            scale=std_sequence,
            size=(config.cem_candidates, config.horizon, action_dim),
        ).astype(np.float32)
        samples = np.clip(samples, action_low, action_high)
        scores = score_action_sequences(
            planner_state,
            samples,
            env_id,
            predict_next,
            observation_for_reward,
        )
        best_idx = int(np.argmax(scores))
        if float(scores[best_idx]) > best_score:
            best_score = float(scores[best_idx])
            best_sequence = samples[best_idx].copy()
        elite_count = max(1, min(config.cem_elites, config.cem_candidates))
        elite_indices = np.argsort(scores)[-elite_count:]
        elites = samples[elite_indices]
        mean_sequence = elites.mean(axis=0)
        std_sequence = np.maximum(elites.std(axis=0), 1e-3)

    return best_sequence[0]


def score_action_sequences(
    planner_state: Any,
    action_sequences: np.ndarray,
    env_id: str,
    predict_next: Callable[[Any, np.ndarray], Any],
    observation_for_reward: Callable[[Any], np.ndarray],
) -> np.ndarray:
    scores = np.zeros(action_sequences.shape[0], dtype=np.float64)
    for candidate_idx, sequence in enumerate(action_sequences):
        state = planner_state
        total = 0.0
        for action in sequence:
            state = predict_next(state, action)
            predicted_observation = observation_for_reward(state)
            total += mujoco_planning_reward_proxy(env_id, predicted_observation, action)
        scores[candidate_idx] = total
    return scores


def mujoco_planning_reward_proxy(
    env_id: str,
    predicted_observation: np.ndarray,
    action: np.ndarray,
) -> float:
    env = env_id.lower()
    state = np.asarray(predicted_observation, dtype=np.float32).reshape(-1)
    action_cost = float(np.sum(np.square(action)))
    if "halfcheetah" in env:
        return _safe_state_value(state, 8) - 0.1 * action_cost
    if "walker2d" in env:
        height = _safe_state_value(state, 0)
        angle = _safe_state_value(state, 1)
        healthy = 1.0 if 0.8 <= height <= 2.0 and -1.0 <= angle <= 1.0 else -1.0
        return _safe_state_value(state, 8) + healthy - 0.001 * action_cost
    if "hopper" in env:
        height = _safe_state_value(state, 0)
        angle = _safe_state_value(state, 1)
        healthy = 1.0 if height > 0.7 and abs(angle) < 0.4 else -1.0
        return _safe_state_value(state, 5) + healthy - 0.001 * action_cost
    if "inverteddoublependulum" in env:
        return -float(np.sum(np.square(state[: min(5, state.shape[0])]))) - 0.001 * action_cost
    if "invertedpendulum" in env:
        x = _safe_state_value(state, 0)
        theta = _safe_state_value(state, 1)
        theta_dot = _safe_state_value(state, 3)
        return 1.0 - theta * theta - 0.1 * x * x - 0.01 * theta_dot * theta_dot - 0.001 * action_cost
    if "reacher" in env:
        tail = state[-3:] if state.shape[0] >= 3 else state
        return -float(np.linalg.norm(tail)) - 0.01 * action_cost
    return -float(np.linalg.norm(state)) - 0.001 * action_cost


def _safe_state_value(state: np.ndarray, index: int) -> float:
    if index < 0 or index >= state.shape[0]:
        return 0.0
    return float(state[index])


def _action_bounds(env: gym.Env) -> tuple[np.ndarray, np.ndarray]:
    low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    low = np.where(np.isfinite(low), low, -1.0).astype(np.float32)
    high = np.where(np.isfinite(high), high, 1.0).astype(np.float32)
    return low, high


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _std(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def sample_rollout_action(env: gym.Env, action_policy: str) -> np.ndarray:
    if action_policy == "random":
        return np.asarray(env.action_space.sample(), dtype=np.float32)
    if action_policy == "zero":
        shape = getattr(env.action_space, "shape", None)
        if shape is None:
            raise ValueError("zero action policy requires a Box-like action space")
        return np.zeros(shape, dtype=np.float32)
    raise ValueError("rollout action policy must be 'zero' or 'random'")


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


def parse_int_list(raw: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not values:
        raise ValueError("list must not be empty")
    return values


if __name__ == "__main__":
    main()
