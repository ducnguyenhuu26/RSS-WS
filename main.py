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
    IslandSearchConfig,
    KinematicPositionLaw,
    LLMLawSynthesisConfig,
    LLMSymbolicLawSynthesizer,
    MuJoCoCollectionConfig,
    ProgramResidualTrainerConfig,
    ProgramResidualWorldModel,
    DeltaGateMLP,
    DiagonalVarianceMLP,
    ResidualMLP,
    ResidualODE,
    SymbolicProgram,
    TransitionBatch,
    build_neural_ensemble_world_model,
    collect_mujoco_dataset,
    fit_neural_ensemble,
    fit_supervised,
    build_law_graph,
    synthesize_with_island_search,
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

    problem = str(cfg.get("probllem", None) or cfg.problem)
    model_name = normalize_model_name(str(cfg.model))
    output_path = build_output_path(cfg, problem, model_name)
    if bool(cfg.get("skip_existing", False)) and output_path.exists():
        print(f"skipping existing output {output_path}")
        return

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

    if is_pets_ensemble_model(model_name):
        run_pets_ensemble_baseline(
            cfg=cfg,
            problem=problem,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            output_path=output_path,
        )
        return

    if is_dreamer_v3_model(model_name):
        raise NotImplementedError(
            "Dreamer V3 is an external baseline in this repo. Generate or import "
            "Dreamer V3 result JSON files with model='dreamer_v3' and the same "
            "score/reward schema, then include them in the formatter input."
        )

    symbolic_source = symbolic_source_for_model(model_name)
    program, llm_code, llm_usage, island_search_summary = build_program_from_config(
        cfg=cfg,
        problem=problem,
        model_name=model_name,
        symbolic_source=symbolic_source,
        train_dataset=train_dataset,
    )
    law_graph_summary = attach_law_graph_budget(
        cfg=cfg,
        problem=problem,
        program=program,
        train_dataset=train_dataset,
    )
    trains_neural_residual = model_uses_neural_residual(model_name)
    if trains_neural_residual:
        if model_uses_ode_residual(model_name):
            residual = ResidualODE(
                state_dim=train_dataset.state_dim,
                action_dim=train_dataset.action_dim,
                hidden_sizes=tuple(int(size) for size in cfg.hidden_sizes),
                transition_dt=resolve_mujoco_dt(problem, cfg.llm.dt),
                ode_steps=int(cfg.get("ode", {}).get("steps", 4)),
            )
        else:
            residual = ResidualMLP(
                state_dim=train_dataset.state_dim,
                action_dim=train_dataset.action_dim,
                hidden_sizes=tuple(int(size) for size in cfg.hidden_sizes),
            )
    else:
        residual = ZeroResidual()
    variance_model = (
        DiagonalVarianceMLP(
            state_dim=train_dataset.state_dim,
            action_dim=train_dataset.action_dim,
            hidden_sizes=tuple(int(size) for size in cfg.hidden_sizes),
        )
        if model_uses_probabilistic_head(model_name)
        else None
    )
    uses_symbolic_gate = model_uses_symbolic_gate(model_name)
    gate_model = (
        DeltaGateMLP(
            state_dim=train_dataset.state_dim,
            action_dim=train_dataset.action_dim,
            hidden_sizes=tuple(int(size) for size in cfg.hidden_sizes),
            initial_logit=float(cfg.get("gate", {}).get("initial_logit", -3.0)),
            temperature=float(cfg.get("gate", {}).get("temperature", 2.0)),
        )
        if uses_symbolic_gate
        else None
    )
    model = ProgramResidualWorldModel(
        state_dim=train_dataset.state_dim,
        action_dim=train_dataset.action_dim,
        program=program,
        residual_model=residual,
        variance_model=variance_model,
        gate_model=gate_model,
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
    if model_has_trainable_parameters(model):
        history = fit_supervised(
            model=model,
            batches=batches,
            config=ProgramResidualTrainerConfig(
                learning_rate=float(cfg.learning_rate),
                residual_l2_weight=float(cfg.residual_l2_weight),
                symbolic_l1_weight=float(
                    cfg.get("symbolic", {}).get("l1_weight", 1e-3)
                ),
                use_nll_loss=bool(
                    cfg.get("probabilistic", {}).get("use_nll_loss", True)
                ),
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
    symbolic_metrics = evaluate_optional_symbolic_diagnostics(
        cfg=cfg,
        problem=problem,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
    )
    results = {
        "problem": problem,
        "model": model_name,
        "symbolic_source": symbolic_source,
        "neural_residual": trains_neural_residual,
        "residual_backbone": residual_backbone_name(model_name),
        "symbolic_gate": uses_symbolic_gate,
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
        "island_search": island_search_summary,
        "law_graph": law_graph_summary,
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
        case "answer" | "answer_mlp" | "program_only":
            return "llm"
        case "symbolic_neural":
            return "standard"
        case "neural" | "neural_mlp":
            return "empty"
        case _:
            raise ValueError(
                "model must be one of: answer, onelife, pets_ensemble, "
                "dreamer_v3, neural, neural_mlp, program_only, "
                "symbolic_neural, answer_mlp"
            )


def model_uses_neural_residual(model_name: str) -> bool:
    return model_name in {
        "answer",
        "answer_mlp",
        "neural",
        "neural_mlp",
        "symbolic_neural",
    }


def model_uses_symbolic_gate(model_name: str) -> bool:
    return model_name in {"answer", "answer_mlp"}


def model_uses_island_search(model_name: str) -> bool:
    return model_name in {"answer", "answer_mlp"}


def model_uses_ode_residual(model_name: str) -> bool:
    return model_name in {"answer", "neural", "symbolic_neural"}


def model_uses_probabilistic_head(model_name: str) -> bool:
    return model_uses_ode_residual(model_name)


def model_uses_trainable_symbolic(model_name: str) -> bool:
    return model_name in {"answer", "answer_mlp", "program_only", "symbolic_neural"}


def residual_backbone_name(model_name: str) -> str:
    if not model_uses_neural_residual(model_name):
        return "none"
    if model_uses_ode_residual(model_name):
        return "ode"
    return "mlp"


def normalize_model_name(model_name: str) -> str:
    raw = str(model_name).strip()
    lowered = raw.lower()
    aliases = {
        "answer": "answer",
        "answer_mlp": "answer_mlp",
        "answer-mlp": "answer_mlp",
        "ans_mlp": "answer_mlp",
        "ans-mlp": "answer_mlp",
        "neural_ode": "neural",
        "neural-ode": "neural",
        "neural_mlp": "neural_mlp",
        "neural-mlp": "neural_mlp",
        "dreamer": "dreamer_v3",
        "dreamerv3": "dreamer_v3",
        "dreamer-v3": "dreamer_v3",
    }
    return aliases.get(lowered, raw)


def is_discrete_symbolic_model(model_name: str) -> bool:
    return model_name in {
        "discrete_symbolic",
        "symbolic_discrete",
        "onelife_standard",
    }


def is_pets_ensemble_model(model_name: str) -> bool:
    return model_name in {
        "pets",
        "pets_ensemble",
        "neural_ensemble",
        "ensemble",
    }


def is_dreamer_v3_model(model_name: str) -> bool:
    return str(model_name).lower() in {
        "dreamer_v3",
        "dreamerv3",
        "dreamer",
    }


def build_program_from_config(
    cfg: DictConfig,
    problem: str,
    model_name: str,
    symbolic_source: str,
    train_dataset,
) -> tuple[SymbolicProgram, str | None, dict[str, int], dict[str, object]]:
    if symbolic_source == "empty":
        return (
            SymbolicProgram(
                state_dim=train_dataset.state_dim,
                laws=[],
                transition_dt=resolve_mujoco_dt(problem, cfg.llm.dt),
                composition_mode="weighted_product_delta",
                unknown_confidence_threshold=float(
                    cfg.get("symbolic", {}).get("unknown_confidence_threshold", 1e-3)
                ),
                base_delta_precision=float(
                    cfg.get("symbolic", {}).get("base_delta_precision", 1.0)
                ),
            ),
            None,
            zero_llm_usage(),
            {},
        )
    if symbolic_source == "standard":
        return build_standard_symbolic_program(
            state_dim=train_dataset.state_dim,
            dt=resolve_mujoco_dt(problem, cfg.llm.dt),
            composition_mode="weighted_product_delta",
            learn_law_weights=model_uses_trainable_symbolic(model_name),
            initial_law_logit=float(
                cfg.get("symbolic", {}).get("initial_law_logit", -8.0)
            ),
            unknown_confidence_threshold=float(
                cfg.get("symbolic", {}).get("unknown_confidence_threshold", 1e-3)
            ),
            base_delta_precision=float(
                cfg.get("symbolic", {}).get("base_delta_precision", 1.0)
            ),
        ), None, zero_llm_usage(), {}

    if cfg.llm.provider == "openai":
        llm_params = OpenAILiteLlmParams(model_slug=str(cfg.llm.model_slug))
    elif cfg.llm.provider == "gemini":
        llm_params = GeminiLiteLlmParams(model_slug=str(cfg.llm.model_slug))
    else:
        raise ValueError("llm.provider must be openai or gemini")

    llm_tracker = LLMCallTracker()
    synthesizer = LLMSymbolicLawSynthesizer(
        llm_params=llm_params,
        llm_client=llm_tracker.client(),
    )
    law_config = LLMLawSynthesisConfig(
        env_id=problem,
        dt=resolve_mujoco_dt(problem, cfg.llm.dt),
        sample_count=int(cfg.llm.sample_count),
    )
    island_summary: dict[str, object] = {}
    if model_uses_island_search(model_name):
        states, actions, next_states = train_dataset.to_torch()
        search_result = synthesize_with_island_search(
            synthesizer=synthesizer,
            batch=TransitionBatch(
                states=states,
                actions=actions,
                next_states=next_states,
            ),
            config=law_config,
            search_config=build_island_search_config(cfg, env_id=problem),
        )
        bundle = search_result.bundle
        island_summary = dict(search_result.summary)
    else:
        bundle = synthesizer.synthesize_from_mujoco_dataset(
            train_dataset,
            law_config,
        )
    return (
        bundle.build_program(
            state_dim=train_dataset.state_dim,
            transition_dt=resolve_mujoco_dt(problem, cfg.llm.dt),
            composition_mode="weighted_product_delta",
            learn_law_weights=model_uses_trainable_symbolic(model_name),
            initial_law_logit=float(
                cfg.get("symbolic", {}).get("initial_law_logit", -8.0)
            ),
            unknown_confidence_threshold=float(
                cfg.get("symbolic", {}).get("unknown_confidence_threshold", 1e-3)
            ),
            base_delta_precision=float(
                cfg.get("symbolic", {}).get("base_delta_precision", 1.0)
            ),
        ),
        bundle.code,
        llm_tracker.as_dict(),
        island_summary,
    )


def attach_law_graph_budget(
    cfg: DictConfig,
    problem: str,
    program: SymbolicProgram,
    train_dataset,
) -> dict[str, object]:
    if len(program.laws) == 0:
        program.set_dimension_budget(torch.zeros(program.state_dim))
        return {
            "num_concepts": 0,
            "num_laws": 0,
            "num_concept_to_law_edges": 0,
            "num_law_to_dim_edges": 0,
            "dimension_budget": [0.0 for _ in range(program.state_dim)],
            "laws": [],
        }
    states, actions, next_states = train_dataset.to_torch()
    graph_cfg = cfg.get("law_graph", {})
    validation_sample_count = int(
        graph_cfg.get(
            "validation_sample_count",
            cfg.get("island_search", {}).get("validation_sample_count", 512),
        )
    )
    quality_floor = float(graph_cfg.get("quality_floor", 0.05))
    law_graph = build_law_graph(
        laws=tuple(program.laws),
        batch=TransitionBatch(
            states=states,
            actions=actions,
            next_states=next_states,
        ),
        env_id=problem,
        transition_dt=resolve_mujoco_dt(problem, cfg.llm.dt),
        validation_sample_count=validation_sample_count,
        quality_floor=quality_floor,
    )
    program.set_dimension_budget(law_graph.dimension_budget)
    return law_graph.to_summary()


def build_standard_symbolic_program(
    state_dim: int,
    dt: float,
    composition_mode: str = "poe_next_state",
    learn_law_weights: bool = False,
    initial_law_logit: float = -8.0,
    unknown_confidence_threshold: float = 1e-6,
    base_delta_precision: float = 1.0,
) -> SymbolicProgram:
    num_positions = max(0, state_dim // 2)
    if num_positions == 0:
        return SymbolicProgram(
            state_dim=state_dim,
            laws=[],
            unknown_confidence_threshold=unknown_confidence_threshold,
            transition_dt=dt,
            composition_mode=composition_mode,
            learn_law_weights=learn_law_weights,
            initial_law_logit=initial_law_logit,
            base_delta_precision=base_delta_precision,
        )
    position_indices = tuple(range(num_positions))
    velocity_indices = tuple(range(num_positions, 2 * num_positions))
    return SymbolicProgram(
        state_dim=state_dim,
        unknown_confidence_threshold=unknown_confidence_threshold,
        transition_dt=dt,
        composition_mode=composition_mode,
        learn_law_weights=learn_law_weights,
        initial_law_logit=initial_law_logit,
        base_delta_precision=base_delta_precision,
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


def model_has_trainable_parameters(model: torch.nn.Module) -> bool:
    return any(parameter.requires_grad for parameter in model.parameters())


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


def run_pets_ensemble_baseline(
    cfg: DictConfig,
    problem: str,
    train_dataset,
    test_dataset,
    output_path: Path,
) -> None:
    device = resolve_torch_device(cfg.get("device", "auto"))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(cfg.seed))
    pets_cfg = cfg.get("pets", {})
    ensemble_size = int(pets_cfg.get("ensemble_size", 5))
    bootstrap = bool(pets_cfg.get("bootstrap", True))
    model = build_neural_ensemble_world_model(
        state_dim=train_dataset.state_dim,
        action_dim=train_dataset.action_dim,
        hidden_sizes=tuple(int(size) for size in cfg.hidden_sizes),
        ensemble_size=ensemble_size,
    ).to(device)
    histories = fit_neural_ensemble(
        model=model,
        dataset=train_dataset,
        batch_size=int(cfg.batch_size),
        config=ProgramResidualTrainerConfig(
            learning_rate=float(cfg.learning_rate),
            residual_l2_weight=float(cfg.residual_l2_weight),
        ),
        num_epochs=int(cfg.epochs),
        seed=int(cfg.seed),
        device=device,
        bootstrap=bootstrap,
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
        rollout_action_policy=str(cfg.rollout.action_policy),
        rollout_stop_on_done=bool(cfg.rollout.stop_on_done),
        planner_config=build_planner_config(cfg),
    )
    symbolic_metrics = evaluate_optional_symbolic_diagnostics(
        cfg=cfg,
        problem=problem,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
    )
    final_losses = [history[-1].loss for history in histories if history]
    results = {
        "problem": problem,
        "model": "pets_ensemble",
        "symbolic_source": "empty",
        "neural_residual": True,
        "symbolic_gate": False,
        "neural_ensemble": True,
        "ensemble_size": ensemble_size,
        "bootstrap": bootstrap,
        "residual_correction": "all_dimensions",
        "seed": int(cfg.seed),
        **runtime_metadata(device),
        "llm_calls": 0,
        "llm_usage": zero_llm_usage(),
        "train_steps": int(cfg.train_steps),
        "test_steps": int(cfg.test_steps),
        "state_dim": train_dataset.state_dim,
        "action_dim": train_dataset.action_dim,
        "final_train_loss": float(np.mean(final_losses)) if final_losses else None,
        "member_final_train_losses": final_losses,
        "score": score_payload_from_metrics(continuous_metrics),
        "reward": reward_payload_from_metrics(continuous_metrics),
        "program_residual": continuous_metrics,
        "symbolic_baselines": symbolic_metrics,
        "island_search": {},
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"wrote {output_path}")


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
    symbolic_metrics = evaluate_optional_symbolic_diagnostics(
        cfg=cfg,
        problem=problem,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
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


def evaluate_optional_symbolic_diagnostics(
    cfg: DictConfig,
    problem: str,
    train_dataset,
    test_dataset,
) -> dict[str, object]:
    diagnostics_cfg = cfg.get("diagnostics", {})
    if not bool(diagnostics_cfg.get("symbolic_baselines", False)):
        return {}
    return evaluate_symbolic_baselines(
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


def build_planner_config(cfg: DictConfig) -> PlannerEvaluationConfig:
    planning_cfg = cfg.get("planning", {})
    return PlannerEvaluationConfig(
        enabled=bool(planning_cfg.get("enabled", False)),
        planners=parse_planner_names(planning_cfg.get("planners", ["cem_pec_mpc"])),
        num_episodes=int(planning_cfg.get("num_episodes", 2)),
        max_episode_steps=int(planning_cfg.get("max_episode_steps", 200)),
        horizon=int(planning_cfg.get("horizon", 5)),
        cem_candidates=int(planning_cfg.get("cem_candidates", 64)),
        cem_elites=int(planning_cfg.get("cem_elites", 8)),
        cem_iterations=int(planning_cfg.get("cem_iterations", 2)),
        state_ood_weight=float(planning_cfg.get("state_ood_weight", 0.0)),
        action_ood_weight=float(planning_cfg.get("action_ood_weight", 0.0)),
        disagreement_weight=float(planning_cfg.get("disagreement_weight", 0.0)),
        ensemble_variance_weight=float(
            planning_cfg.get("ensemble_variance_weight", 0.0)
        ),
        ood_z_clip=float(planning_cfg.get("ood_z_clip", 3.0)),
    )


def parse_planner_names(raw_planners) -> tuple[str, ...]:
    if raw_planners is None:
        return ("cem_pec_mpc",)
    if isinstance(raw_planners, str):
        planners = tuple(
            planner.strip()
            for planner in raw_planners.split(",")
            if planner.strip()
        )
    else:
        planners = tuple(str(planner).strip() for planner in raw_planners)
    if not planners:
        return ("cem_pec_mpc",)
    allowed = {"cem_mpc", "cem_pec_mpc"}
    unknown = sorted(set(planners) - allowed)
    if unknown:
        raise ValueError(
            "planning.planners must contain only 'cem_mpc' and/or "
            f"'cem_pec_mpc', got {unknown}"
        )
    return planners


def build_island_search_config(cfg: DictConfig, env_id: str | None = None) -> IslandSearchConfig:
    island_cfg = cfg.get("island_search", {})
    return IslandSearchConfig(
        env_id=env_id,
        candidates_per_niche=int(island_cfg.get("candidates_per_niche", 1)),
        generations=int(island_cfg.get("generations", 1)),
        island_size=int(island_cfg.get("island_size", 4)),
        elite_per_island=int(island_cfg.get("elite_per_island", 1)),
        migration_interval=int(island_cfg.get("migration_interval", 1)),
        migrants_per_island=int(island_cfg.get("migrants_per_island", 1)),
        max_laws_per_program=int(island_cfg.get("max_laws_per_program", 4)),
        validation_sample_count=int(island_cfg.get("validation_sample_count", 512)),
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
