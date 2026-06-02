from __future__ import annotations

import json
import os
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import get_original_cwd
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from onelife.litellm_utils import GeminiLiteLlmParams, OpenAILiteLlmParams
from onelife.mujoco_onelife_llm import (
    LLMOneLifeMuJoCoSynthesizer,
    LLMOneLifeSynthesisConfig,
    evaluate_onelife_llm_baseline,
)
from onelife.mujoco_symbolic_adapter import (
    MuJoCoDiscretizer,
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
from scripts.run_mujoco_experiment import (
    evaluate_program_residual,
    evaluate_symbolic_baselines,
    make_batches,
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
    program, llm_code = build_program_from_config(
        cfg=cfg,
        problem=problem,
        symbolic_source=symbolic_source,
        train_dataset=train_dataset,
    )
    residual = ResidualMLP(
        state_dim=train_dataset.state_dim,
        action_dim=train_dataset.action_dim,
        hidden_sizes=tuple(int(size) for size in cfg.hidden_sizes),
    )
    model = ProgramResidualWorldModel(
        state_dim=train_dataset.state_dim,
        action_dim=train_dataset.action_dim,
        program=program,
        residual_model=residual,
    )
    batches = make_batches(
        train_dataset.iter_torch_batches(
            batch_size=int(cfg.batch_size),
            shuffle=True,
            seed=int(cfg.seed),
        )
    )
    history = fit_supervised(
        model=model,
        batches=batches,
        config=ProgramResidualTrainerConfig(
            learning_rate=float(cfg.learning_rate),
            residual_l2_weight=float(cfg.residual_l2_weight),
        ),
        num_epochs=int(cfg.epochs),
    )

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
    )
    symbolic_metrics = evaluate_symbolic_baselines(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        state_bins=int(cfg.discretization.state_bins),
        action_bins=int(cfg.discretization.action_bins),
    )
    results = {
        "problem": problem,
        "model": model_name,
        "symbolic_source": symbolic_source,
        "seed": int(cfg.seed),
        "train_steps": int(cfg.train_steps),
        "test_steps": int(cfg.test_steps),
        "state_dim": train_dataset.state_dim,
        "action_dim": train_dataset.action_dim,
        "final_train_loss": history[-1].loss if history else None,
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
        case "ours":
            return "llm"
        case "residual" | "residual_only" | "empty" | "no_llm":
            return "empty"
        case _:
            raise ValueError(
                "model must be one of: ours, onelife, residual, residual_only, empty, no_llm"
            )


def build_program_from_config(
    cfg: DictConfig,
    problem: str,
    symbolic_source: str,
    train_dataset,
) -> tuple[SymbolicProgram, str | None]:
    if symbolic_source == "empty":
        return SymbolicProgram(state_dim=train_dataset.state_dim, laws=[]), None

    if cfg.llm.provider == "openai":
        llm_params = OpenAILiteLlmParams(model_slug=str(cfg.llm.model_slug))
    elif cfg.llm.provider == "gemini":
        llm_params = GeminiLiteLlmParams(model_slug=str(cfg.llm.model_slug))
    else:
        raise ValueError("llm.provider must be openai or gemini")

    bundle = LLMSymbolicLawSynthesizer(
        llm_params=llm_params
    ).synthesize_from_mujoco_dataset(
        train_dataset,
        LLMLawSynthesisConfig(
            env_id=problem,
            dt=float(cfg.llm.dt),
            sample_count=int(cfg.llm.sample_count),
        ),
    )
    return bundle.build_program(state_dim=train_dataset.state_dim), bundle.code


def run_onelife_llm_baseline(
    cfg: DictConfig,
    problem: str,
    train_dataset,
    test_dataset,
    output_path: Path,
) -> None:
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

    bundle = LLMOneLifeMuJoCoSynthesizer(
        llm_params=llm_params
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
    symbolic_metrics = evaluate_symbolic_baselines(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        state_bins=int(cfg.discretization.state_bins),
        action_bins=int(cfg.discretization.action_bins),
    )
    results = {
        "problem": problem,
        "model": "onelife",
        "symbolic_source": "llm",
        "seed": int(cfg.seed),
        "train_steps": int(cfg.train_steps),
        "test_steps": int(cfg.test_steps),
        "state_dim": train_dataset.state_dim,
        "action_dim": train_dataset.action_dim,
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
