from __future__ import annotations

import json
import os
from pathlib import Path

import hydra
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from hydra.utils import get_original_cwd
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from onelife.litellm_utils import (
    GeminiLiteLlmParams,
    LLMCallTracker,
    OpenAILiteLlmParams,
    zero_llm_usage,
)
from onelife.mujoco_onelife_llm import (
    LLMOneLifeMuJoCoSynthesizer,
    LLMOneLifeSynthesisConfig,
    evaluate_onelife_llm_baseline,
)
from onelife.mujoco_symbolic_adapter import (
    MuJoCoDiscretizer,
)
from onelife.program_residual import (
    KinematicPositionLaw,
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
from scripts.run_mujoco_experiment import (
    PlannerEvaluationConfig,
    evaluate_binned_model_r2,
    evaluate_binned_planner_rewards,
    evaluate_program_residual,
    evaluate_symbolic_baselines,
    evaluate_symbolic_open_loop_rollouts,
    make_batches,
    reward_payload_from_metrics,
    score_payload_from_metrics,
)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    load_env_file(Path(get_original_cwd()) / ".env")
    if not cfg.verbose:
        logger.disable("onelife")

    problem = str(cfg.probllem or cfg.problem)
    model_name = str(cfg.model)
    output_path = build_output_path(cfg, problem, model_name)

    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    device = resolve_torch_device(cfg.get("device", "auto"))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(cfg.seed))

    train_dataset = collect_mujoco_dataset(
        config=MuJoCoCollectionConfig(
            env_id=problem,
            num_steps=int(cfg.train_steps),
            seed=int(cfg.seed),
        )
    )
    test_dataset = collect_mujoco_dataset(
        config=MuJoCoCollectionConfig(
            env_id=problem,
            num_steps=int(cfg.test_steps),
            seed=int(cfg.seed) + 1,
        )
    )

    if is_discrete_symbolic_model(model_name):
        run_discrete_symbolic_baseline(
            cfg=cfg,
            problem=problem,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            output_path=output_path,
        )
        return

    if model_name == "onelife":
        run_onelife_llm_baseline(
            cfg=cfg,
            problem=problem,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            output_path=output_path,
        )
        return

    symbolic_source = symbolic_source_for_model(model_name)
    program, llm_code, llm_usage = build_program_from_config(
        cfg=cfg,
        problem=problem,
        symbolic_source=symbolic_source,
        train_dataset=train_dataset,
    )
    trains_neural_residual = model_uses_neural_residual(model_name)
    if trains_neural_residual:
        residual = ResidualMLP(
            state_dim=train_dataset.state_dim,
            action_dim=train_dataset.action_dim,
            hidden_sizes=tuple(int(size) for size in cfg.hidden_sizes),
        )
    else:
        residual = ZeroResidual()
    model = ProgramResidualWorldModel(
        state_dim=train_dataset.state_dim,
        action_dim=train_dataset.action_dim,
        program=program,
        residual_model=residual,
        apply_unknown_mask=not trains_neural_residual,
    ).to(device)
    batches = [
        batch.to(device)
        for batch in make_batches(
            train_dataset.iter_torch_batches(
                batch_size=int(cfg.batch_size),
                shuffle=True,
                seed=int(cfg.seed),
            )
        )
    ]
    if trains_neural_residual:
        history = fit_supervised(
            model=model,
            batches=batches,
            config=ProgramResidualTrainerConfig(
                learning_rate=float(cfg.learning_rate),
                residual_l2_weight=float(cfg.residual_l2_weight),
            ),
            num_epochs=int(cfg.epochs),
        )
    else:
        history = []

    continuous_metrics = evaluate_program_residual(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        env_id=problem,
        seed=int(cfg.seed) + 10_000,
        state_bins=int(cfg.discretization.state_bins),
        action_bins=int(cfg.discretization.action_bins),
        rollout_horizons=tuple(int(horizon) for horizon in cfg.rollout.horizons),
        rollout_num_rollouts=int(cfg.rollout.num_rollouts),
        rollout_action_policy=str(cfg.rollout.action_policy),
        rollout_stop_on_done=bool(cfg.rollout.stop_on_done),
        planner_config=build_planner_config(cfg),
    )
    symbolic_metrics = evaluate_symbolic_baselines(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        state_bins=int(cfg.discretization.state_bins),
        action_bins=int(cfg.discretization.action_bins),
        env_id=problem,
        seed=int(cfg.seed) + 20_000,
        rollout_horizons=tuple(int(horizon) for horizon in cfg.rollout.horizons),
        rollout_num_rollouts=int(cfg.rollout.num_rollouts),
        rollout_action_policy=str(cfg.rollout.action_policy),
        rollout_stop_on_done=bool(cfg.rollout.stop_on_done),
    )
    results = {
        "problem": problem,
        "model": model_name,
        "symbolic_source": symbolic_source,
        "neural_residual": trains_neural_residual,
        "residual_correction": "all_dimensions"
        if trains_neural_residual
        else "none",
        "seed": int(cfg.seed),
        **runtime_metadata(device),
        "llm_calls": int(llm_usage["calls"]),
        "llm_usage": llm_usage,
        "train_steps": int(cfg.train_steps),
        "test_steps": int(cfg.test_steps),
        "state_dim": train_dataset.state_dim,
        "action_dim": train_dataset.action_dim,
        "final_train_loss": history[-1].loss if history else None,
        "score": score_payload_from_metrics(continuous_metrics),
        "reward": reward_payload_from_metrics(continuous_metrics),
        "program_residual": continuous_metrics,
        "symbolic_baselines": symbolic_metrics,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if llm_code is not None:
        law_path = output_path.with_suffix(".laws.py")
        law_path.write_text(llm_code, encoding="utf-8")
        results["llm_law_path"] = str(law_path)
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(json.dumps(results, indent=2))
    print(f"wrote {output_path}")


def symbolic_source_for_model(model_name: str) -> str:
    match model_name:
        case "ours" | "program_only" | "llm_symbolic" | "symbolic_llm":
            return "llm"
        case (
            "symbolic"
            | "standard_symbolic"
            | "symbolic_only"
            | "no_llm_symbolic"
            | "symbolic_neural"
            | "standard_symbolic_neural"
            | "no_llm_symbolic_neural"
        ):
            return "standard"
        case "neural" | "residual" | "residual_only" | "empty" | "no_llm":
            return "empty"
        case _:
            raise ValueError(
                "model must be one of: ours, program_only, neural, symbolic, "
                "symbolic_neural, onelife, discrete_symbolic"
            )


def model_uses_neural_residual(model_name: str) -> bool:
    return model_name in {
        "ours",
        "neural",
        "symbolic_neural",
        "standard_symbolic_neural",
        "no_llm_symbolic_neural",
        "residual",
        "residual_only",
        "empty",
        "no_llm",
    }


def is_discrete_symbolic_model(model_name: str) -> bool:
    return model_name in {
        "discrete_symbolic",
        "symbolic_discrete",
        "onelife_standard",
    }


def build_program_from_config(
    cfg: DictConfig,
    problem: str,
    symbolic_source: str,
    train_dataset,
) -> tuple[SymbolicProgram, str | None, dict[str, int]]:
    if symbolic_source == "empty":
        return (
            SymbolicProgram(state_dim=train_dataset.state_dim, laws=[]),
            None,
            zero_llm_usage(),
        )
    if symbolic_source == "standard":
        return build_standard_symbolic_program(
            state_dim=train_dataset.state_dim,
            dt=resolve_mujoco_dt(problem, cfg.llm.dt),
        ), None, zero_llm_usage()

    if cfg.llm.provider == "openai":
        llm_params = OpenAILiteLlmParams(model_slug=str(cfg.llm.model_slug))
    elif cfg.llm.provider == "gemini":
        llm_params = GeminiLiteLlmParams(model_slug=str(cfg.llm.model_slug))
    else:
        raise ValueError("llm.provider must be openai or gemini")

    llm_tracker = LLMCallTracker()
    bundle = LLMSymbolicLawSynthesizer(
        llm_params=llm_params,
        llm_client=llm_tracker.client(),
    ).synthesize_from_mujoco_dataset(
        train_dataset,
        LLMLawSynthesisConfig(
            env_id=problem,
            dt=resolve_mujoco_dt(problem, cfg.llm.dt),
            sample_count=int(cfg.llm.sample_count),
        ),
    )
    return (
        bundle.build_program(state_dim=train_dataset.state_dim),
        bundle.code,
        llm_tracker.as_dict(),
    )


def build_standard_symbolic_program(state_dim: int, dt: float) -> SymbolicProgram:
    num_positions = max(0, state_dim // 2)
    if num_positions == 0:
        return SymbolicProgram(state_dim=state_dim, laws=[])
    position_indices = tuple(range(num_positions))
    velocity_indices = tuple(range(num_positions, 2 * num_positions))
    return SymbolicProgram(
        state_dim=state_dim,
        laws=[
            KinematicPositionLaw(
                position_indices=position_indices,
                velocity_indices=velocity_indices,
                dt=dt,
                confidence=1.0,
                name="StandardKinematicPositionLaw",
            )
        ],
    )


def resolve_mujoco_dt(problem: str, raw_dt) -> float:
    if str(raw_dt).lower() != "auto":
        return float(raw_dt)
    env = gym.make(problem)
    try:
        dt = getattr(env.unwrapped, "dt", None)
        return float(dt) if dt is not None else 0.05
    finally:
        env.close()


class ZeroResidual(nn.Module):
    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        program_next_state: torch.Tensor,
        unknown_mask: torch.Tensor,
    ) -> torch.Tensor:
        return torch.zeros_like(states)


def run_discrete_symbolic_baseline(
    cfg: DictConfig,
    problem: str,
    train_dataset,
    test_dataset,
    output_path: Path,
) -> None:
    device = resolve_torch_device(cfg.get("device", "auto"))
    symbolic_metrics = evaluate_symbolic_baselines(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        state_bins=int(cfg.discretization.state_bins),
        action_bins=int(cfg.discretization.action_bins),
        env_id=problem,
        seed=int(cfg.seed) + 20_000,
        rollout_horizons=tuple(int(horizon) for horizon in cfg.rollout.horizons),
        rollout_num_rollouts=int(cfg.rollout.num_rollouts),
        rollout_action_policy=str(cfg.rollout.action_policy),
        rollout_stop_on_done=bool(cfg.rollout.stop_on_done),
    )
    results = {
        "problem": problem,
        "model": str(cfg.model),
        "symbolic_source": "standard_no_llm",
        "neural_residual": False,
        "seed": int(cfg.seed),
        **runtime_metadata(device),
        "llm_calls": 0,
        "llm_usage": zero_llm_usage(),
        "train_steps": int(cfg.train_steps),
        "test_steps": int(cfg.test_steps),
        "state_dim": train_dataset.state_dim,
        "action_dim": train_dataset.action_dim,
        "symbolic_baselines": symbolic_metrics,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"wrote {output_path}")


def run_onelife_llm_baseline(
    cfg: DictConfig,
    problem: str,
    train_dataset,
    test_dataset,
    output_path: Path,
) -> None:
    device = resolve_torch_device(cfg.get("device", "auto"))
    discretizer = MuJoCoDiscretizer.fit(
        train_dataset,
        state_bins=int(cfg.discretization.state_bins),
        action_bins=int(cfg.discretization.action_bins),
    )
    if cfg.llm.provider == "openai":
        llm_params = OpenAILiteLlmParams(model_slug=str(cfg.llm.model_slug))
    elif cfg.llm.provider == "gemini":
        llm_params = GeminiLiteLlmParams(model_slug=str(cfg.llm.model_slug))
    else:
        raise ValueError("llm.provider must be openai or gemini")

    llm_tracker = LLMCallTracker()
    bundle = LLMOneLifeMuJoCoSynthesizer(
        llm_params=llm_params,
        llm_client=llm_tracker.client(),
    ).synthesize_from_dataset(
        train_dataset,
        discretizer,
        LLMOneLifeSynthesisConfig(
            env_id=problem,
            sample_count=int(cfg.llm.sample_count),
        ),
    )
    onelife_llm_metrics = evaluate_onelife_llm_baseline(
        laws=bundle.laws,
        discretizer=discretizer,
        test_dataset=test_dataset,
    )
    onelife_model = bundle.build_law_mixture(discretizer)
    onelife_llm_metrics.update(
        evaluate_binned_model_r2(
            model=onelife_model,
            discretizer=discretizer,
            test_dataset=test_dataset,
        )
    )
    onelife_llm_metrics.update(
        evaluate_symbolic_open_loop_rollouts(
            model=onelife_model,
            discretizer=discretizer,
            env_id=problem,
            seed=int(cfg.seed) + 30_000,
            horizons=tuple(int(horizon) for horizon in cfg.rollout.horizons),
            num_rollouts=int(cfg.rollout.num_rollouts),
            action_policy=str(cfg.rollout.action_policy),
            stop_on_done=bool(cfg.rollout.stop_on_done),
        )
    )
    planner_config = build_planner_config(cfg)
    if planner_config.enabled:
        onelife_llm_metrics.update(
            evaluate_binned_planner_rewards(
                model=onelife_model,
                discretizer=discretizer,
                env_id=problem,
                seed=int(cfg.seed) + 30_000 + planner_config.action_policy_seed_offset,
                config=planner_config,
            )
        )
    symbolic_metrics = evaluate_symbolic_baselines(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        state_bins=int(cfg.discretization.state_bins),
        action_bins=int(cfg.discretization.action_bins),
        env_id=problem,
        seed=int(cfg.seed) + 20_000,
        rollout_horizons=tuple(int(horizon) for horizon in cfg.rollout.horizons),
        rollout_num_rollouts=int(cfg.rollout.num_rollouts),
        rollout_action_policy=str(cfg.rollout.action_policy),
        rollout_stop_on_done=bool(cfg.rollout.stop_on_done),
    )
    llm_usage = llm_tracker.as_dict()
    results = {
        "problem": problem,
        "model": "onelife",
        "symbolic_source": "llm",
        "seed": int(cfg.seed),
        **runtime_metadata(device),
        "llm_calls": int(llm_usage["calls"]),
        "llm_usage": llm_usage,
        "train_steps": int(cfg.train_steps),
        "test_steps": int(cfg.test_steps),
        "state_dim": train_dataset.state_dim,
        "action_dim": train_dataset.action_dim,
        "score": score_payload_from_metrics(onelife_llm_metrics),
        "reward": reward_payload_from_metrics(onelife_llm_metrics),
        "onelife_llm": onelife_llm_metrics,
        "symbolic_baselines": symbolic_metrics,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    law_path = output_path.with_suffix(".laws.py")
    law_path.write_text(bundle.code, encoding="utf-8")
    results["llm_law_path"] = str(law_path)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"wrote {output_path}")


def build_output_path(cfg: DictConfig, problem: str, model_name: str) -> Path:
    root = Path(get_original_cwd())
    output_dir = root / str(cfg.output_dir)
    output_name = cfg.output_name
    if output_name is None:
        safe_problem = problem.replace("/", "_")
        output_name = f"{safe_problem}-{model_name}-seed{int(cfg.seed)}.json"
    return output_dir / str(output_name)


def build_planner_config(cfg: DictConfig) -> PlannerEvaluationConfig:
    planning_cfg = cfg.get("planning", {})
    return PlannerEvaluationConfig(
        enabled=bool(planning_cfg.get("enabled", False)),
        num_episodes=int(planning_cfg.get("num_episodes", 2)),
        max_episode_steps=int(planning_cfg.get("max_episode_steps", 200)),
        horizon=int(planning_cfg.get("horizon", 5)),
        random_candidates=int(planning_cfg.get("random_candidates", 64)),
        cem_candidates=int(planning_cfg.get("cem_candidates", 64)),
        cem_elites=int(planning_cfg.get("cem_elites", 8)),
        cem_iterations=int(planning_cfg.get("cem_iterations", 2)),
    )


def resolve_torch_device(raw_device) -> torch.device:
    raw = "auto" if raw_device is None else str(raw_device).strip().lower()
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(raw)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def runtime_metadata(device: torch.device) -> dict[str, object]:
    return {
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else None,
    }


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


if __name__ == "__main__":
    main()
