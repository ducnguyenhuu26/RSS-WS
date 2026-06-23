from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from onelife.mujoco_dataset import MuJoCoTransitions, concatenate_mujoco_transitions

from onelife.duc_wm import (
    BaselineTrainerConfig,
    CaDMWorldModel,
    CaDMWorldModelConfig,
    DUCLLMPriorConfig,
    DUCTrainerConfig,
    DUCWorldModel,
    DUCWorldModelConfig,
    MLPWorldModel,
    MLPWorldModelConfig,
    MuJoCoExtensionConfig,
    PETSWorldModel,
    PETSWorldModelConfig,
    PlanningEvalConfig,
    RewardModel,
    RewardModelConfig,
    RewardTrainerConfig,
    build_duc_mujoco_prior_prompt,
    collect_mujoco_extension_dataset,
    default_mujoco_templates,
    evaluate_baseline_world_model,
    evaluate_cem_mpc,
    evaluate_duc_model,
    fit_baseline_world_model,
    fit_duc_world_model,
    fit_reward_model,
    generic_mechanism_templates,
    load_templates_from_json_file,
    prompt_payload,
    randomize_mechanism_templates,
    remove_unknown_template,
    synthesize_templates_with_llm,
    wrong_mechanism_templates,
)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    configure_runtime(cfg)
    seed = int(cfg.seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = resolve_torch_device(str(cfg.get("device", "auto")))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    problem = str(cfg.problem)
    method = str(cfg.get("model", "duc_wm"))
    output_path = build_output_path(cfg, problem)
    if bool(cfg.get("skip_existing", False)) and output_path.exists():
        print(f"skipping existing output {output_path}")
        return

    train_variants = resolve_variants(
        cfg.mujoco_extension.get("train_variants"),
        fallback=str(cfg.mujoco_extension.variant),
    )
    test_variants = resolve_variants(
        cfg.mujoco_extension.get("test_variants"),
        fallback=str(cfg.mujoco_extension.test_variant),
    )
    train_dataset = collect_mujoco_variants(
        problem=problem,
        total_steps=int(cfg.train_steps),
        seed=seed,
        variants=train_variants,
        num_contexts=int(cfg.mujoco_extension.num_contexts),
        cfg=cfg,
    )
    test_dataset = collect_mujoco_variants(
        problem=problem,
        total_steps=int(cfg.test_steps),
        seed=seed + 10_000,
        variants=test_variants,
        num_contexts=int(cfg.mujoco_extension.test_num_contexts),
        cfg=cfg,
    )

    prior_prompt = build_duc_mujoco_prior_prompt(
        env_id=problem,
        state_dim=train_dataset.state_dim,
        action_dim=train_dataset.action_dim,
    )
    prior_path = cfg.duc.llm_prior.get("json_path")
    should_use_duc_prior = method in DUC_METHODS
    llm_prior_status: dict[str, Any] = {
        "source": "fallback",
        "error": None,
        "raw_response": None,
    }
    if should_use_duc_prior and prior_path:
        templates = load_templates_from_json_file(
            Path(str(prior_path)),
            state_dim=train_dataset.state_dim,
            action_dim=train_dataset.action_dim,
        )
        llm_prior_status["source"] = f"json_path:{prior_path}"
    elif should_use_duc_prior and bool(cfg.duc.llm_prior.get("enabled", False)):
        try:
            templates, raw_response = synthesize_templates_with_llm(
                prior_prompt=prior_prompt,
                state_dim=train_dataset.state_dim,
                action_dim=train_dataset.action_dim,
                config=DUCLLMPriorConfig(
                    provider=str(cfg.duc.llm_prior.provider),
                    model_slug=str(cfg.duc.llm_prior.model_slug),
                    api_key_env=str(cfg.duc.llm_prior.api_key_env),
                    max_tokens=int(cfg.duc.llm_prior.max_tokens),
                ),
            )
            llm_prior_status["source"] = (
                f"llm:{cfg.duc.llm_prior.provider}/{cfg.duc.llm_prior.model_slug}"
            )
            llm_prior_status["raw_response"] = raw_response
        except Exception as exc:
            templates = prior_prompt.fallback_templates
            llm_prior_status["source"] = "fallback_after_llm_error"
            llm_prior_status["error"] = str(exc)
    else:
        templates = prior_prompt.fallback_templates

    if method == "duc_random":
        templates = randomize_mechanism_templates(
            templates=templates,
            state_dim=train_dataset.state_dim,
            action_dim=train_dataset.action_dim,
            seed=seed + 8128,
        )
        llm_prior_status["source"] = f"{llm_prior_status['source']}+random_masks"
    elif method == "duc_no_llm":
        templates = generic_mechanism_templates(
            state_dim=train_dataset.state_dim,
            action_dim=train_dataset.action_dim,
            count=max(1, len(templates) - 1),
            include_unknown=True,
        )
        llm_prior_status["source"] = "generic_no_llm_prior"
    elif method == "duc_wrong_prior":
        templates = wrong_mechanism_templates(templates)
        llm_prior_status["source"] = f"{llm_prior_status['source']}+wrong_rotated_masks"
    elif method == "duc_no_unknown":
        templates = remove_unknown_template(templates)
        llm_prior_status["source"] = f"{llm_prior_status['source']}+no_unknown"

    if method in DUC_METHODS:
        model = DUCWorldModel(
            DUCWorldModelConfig(
                state_dim=train_dataset.state_dim,
                action_dim=train_dataset.action_dim,
                templates=templates,
                hidden_size=int(cfg.duc.hidden_size),
                hidden_layers=int(cfg.duc.hidden_layers),
                history_length=int(cfg.duc.history_length),
                trust_region_delta_min=float(config_get(cfg, "duc.trust_region_delta_min", 0.25)),
                trust_region_delta_range=float(config_get(cfg, "duc.trust_region_delta_range", 1.25)),
            )
        ).to(device)
        maybe_compile_forward(model, cfg)
        history = fit_duc_world_model(
            model=model,
            transitions=train_dataset,
            config=duc_trainer_config(cfg, seed, method),
            device=device,
        )
        metrics = evaluate_duc_model(
            model=model,
            transitions=test_dataset,
            device=device,
            batch_size=int(cfg.eval_batch_size),
            history_length=int(cfg.duc.history_length),
            rollout_horizon=int(cfg.duc.rollout_horizon),
        )
    elif method == "mlp":
        model = MLPWorldModel(
            MLPWorldModelConfig(
                state_dim=train_dataset.state_dim,
                action_dim=train_dataset.action_dim,
                hidden_size=int(config_get(cfg, "baseline.hidden_size", cfg.duc.hidden_size)),
                hidden_layers=int(config_get(cfg, "baseline.hidden_layers", cfg.duc.hidden_layers)),
            )
        ).to(device)
        maybe_compile_forward(model, cfg)
        history = fit_baseline_world_model(
            model=model,
            transitions=train_dataset,
            config=baseline_trainer_config(cfg, seed),
            device=device,
            control_templates=templates,
        )
        metrics = evaluate_baseline_world_model(
            model=model,
            transitions=test_dataset,
            device=device,
            control_templates=templates,
            batch_size=int(cfg.eval_batch_size),
            history_length=int(cfg.duc.history_length),
            rollout_horizon=int(cfg.duc.rollout_horizon),
        )
    elif method == "pets":
        model = PETSWorldModel(
            PETSWorldModelConfig(
                state_dim=train_dataset.state_dim,
                action_dim=train_dataset.action_dim,
                hidden_size=int(config_get(cfg, "baseline.hidden_size", cfg.duc.hidden_size)),
                hidden_layers=int(config_get(cfg, "baseline.hidden_layers", cfg.duc.hidden_layers)),
                ensemble_size=int(config_get(cfg, "baseline.ensemble_size", 5)),
            )
        ).to(device)
        maybe_compile_forward(model, cfg)
        history = fit_baseline_world_model(
            model=model,
            transitions=train_dataset,
            config=baseline_trainer_config(cfg, seed),
            device=device,
            control_templates=templates,
        )
        metrics = evaluate_baseline_world_model(
            model=model,
            transitions=test_dataset,
            device=device,
            control_templates=templates,
            batch_size=int(cfg.eval_batch_size),
            history_length=int(cfg.duc.history_length),
            rollout_horizon=int(cfg.duc.rollout_horizon),
        )
    elif method == "cadm":
        model = CaDMWorldModel(
            CaDMWorldModelConfig(
                state_dim=train_dataset.state_dim,
                action_dim=train_dataset.action_dim,
                history_length=int(cfg.duc.history_length),
                context_dim=int(config_get(cfg, "baseline.context_dim", len(templates))),
                hidden_size=int(config_get(cfg, "baseline.hidden_size", cfg.duc.hidden_size)),
                hidden_layers=int(config_get(cfg, "baseline.hidden_layers", cfg.duc.hidden_layers)),
            )
        ).to(device)
        maybe_compile_forward(model, cfg)
        history = fit_baseline_world_model(
            model=model,
            transitions=train_dataset,
            config=baseline_trainer_config(cfg, seed),
            device=device,
            control_templates=templates,
        )
        metrics = evaluate_baseline_world_model(
            model=model,
            transitions=test_dataset,
            device=device,
            control_templates=templates,
            batch_size=int(cfg.eval_batch_size),
            history_length=int(cfg.duc.history_length),
            rollout_horizon=int(cfg.duc.rollout_horizon),
        )
    else:
        raise ValueError(
            f"unknown model={method!r}; expected one of {sorted(ALL_METHODS)}"
        )

    reward_history: list[dict[str, float]] = []
    if bool(config_get(cfg, "planning.enabled", False)):
        reward_model = RewardModel(
            RewardModelConfig(
                state_dim=train_dataset.state_dim,
                action_dim=train_dataset.action_dim,
                hidden_size=int(config_get(cfg, "reward.hidden_size", cfg.duc.hidden_size)),
                hidden_layers=int(config_get(cfg, "reward.hidden_layers", cfg.duc.hidden_layers)),
            )
        ).to(device)
        reward_history = fit_reward_model(
            model=reward_model,
            transitions=train_dataset,
            config=reward_trainer_config(cfg, seed),
            device=device,
        )
        metrics.update(
            evaluate_cem_mpc(
                dynamics_model=model,
                reward_model=reward_model,
                env_id=problem,
                variants=test_variants,
                seed=seed + 20_000,
                action_smoothing=float(cfg.mujoco_extension.action_smoothing),
                history_length=int(cfg.duc.history_length),
                config=planning_eval_config(cfg),
                device=device,
            )
        )

    payload = {
        "framework": "DUC-WM benchmark",
        "model": method,
        "problem": problem,
        "seed": seed,
        "device": str(device),
        "train_steps": train_dataset.num_steps,
        "test_steps": test_dataset.num_steps,
        "variant": variant_label(train_variants),
        "test_variant": variant_label(test_variants),
        "train_variants": train_variants,
        "test_variants": test_variants,
        "context_names": list(train_dataset.context_names),
        "llm_prior_prompt": prompt_payload(prior_prompt),
        "llm_prior_status": llm_prior_status,
        "mechanisms": [
            {
                "name": template.name,
                "state_indices": list(template.state_indices),
                "action_indices": list(template.action_indices),
                "output_indices": list(template.output_indices),
                "law_type": template.law_type,
                "law_gain": template.law_gain,
                "scale": template.scale,
                "prior_mean": template.prior_mean,
                "prior_std": template.prior_std,
                "prior_confidence": template.prior_confidence,
                "timescale": template.timescale,
                "reward_relevance": template.reward_relevance,
                "description": template.description,
            }
            for template in templates
        ],
        "training": history,
        "reward_training": reward_history,
        "score": metrics,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "score": metrics}, indent=2))


def resolve_torch_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(raw)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return device


def build_output_path(cfg: DictConfig, problem: str) -> Path:
    root = Path(get_original_cwd())
    output_dir = Path(str(cfg.output_dir))
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_name = cfg.get("output_name")
    if output_name:
        return output_dir / str(output_name)
    method = str(cfg.get("model", "duc_wm"))
    safe_problem = problem.replace("/", "_").replace(":", "_")
    safe_method = method.replace("/", "_").replace(":", "_")
    train_variants = resolve_variants(
        cfg.mujoco_extension.get("train_variants"),
        fallback=str(cfg.mujoco_extension.variant),
    )
    test_variants = resolve_variants(
        cfg.mujoco_extension.get("test_variants"),
        fallback=str(cfg.mujoco_extension.test_variant),
    )
    safe_train = safe_label(variant_label(train_variants))
    safe_test = safe_label(variant_label(test_variants))
    return output_dir / (
        f"{safe_method}_{safe_problem}_train-{safe_train}_test-{safe_test}_"
        f"seed{int(cfg.seed)}.json"
    )


def baseline_trainer_config(cfg: DictConfig, seed: int) -> BaselineTrainerConfig:
    return BaselineTrainerConfig(
        epochs=int(cfg.epochs),
        batch_size=int(cfg.batch_size),
        learning_rate=float(cfg.learning_rate),
        history_length=int(cfg.duc.history_length),
        control_weight=float(cfg.duc.control_weight),
        rollout_weight=float(cfg.duc.rollout_weight),
        rollout_horizon=int(cfg.duc.rollout_horizon),
        seed=seed,
        precision=str(config_get(cfg, "runtime.precision", "fp32")),
        preload_to_device=bool(config_get(cfg, "runtime.preload_to_device", False)),
    )


def reward_trainer_config(cfg: DictConfig, seed: int) -> RewardTrainerConfig:
    return RewardTrainerConfig(
        epochs=int(config_get(cfg, "reward.epochs", max(1, int(cfg.epochs) // 3))),
        batch_size=int(config_get(cfg, "reward.batch_size", cfg.batch_size)),
        learning_rate=float(config_get(cfg, "reward.learning_rate", cfg.learning_rate)),
        history_length=int(cfg.duc.history_length),
        seed=seed,
    )


def planning_eval_config(cfg: DictConfig) -> PlanningEvalConfig:
    return PlanningEvalConfig(
        enabled=bool(config_get(cfg, "planning.enabled", False)),
        episodes=int(config_get(cfg, "planning.episodes", 3)),
        max_steps=int(config_get(cfg, "planning.max_steps", 200)),
        horizon=int(config_get(cfg, "planning.horizon", 15)),
        candidates=int(config_get(cfg, "planning.candidates", 256)),
        elites=int(config_get(cfg, "planning.elites", 32)),
        iterations=int(config_get(cfg, "planning.iterations", 3)),
        uncertainty_weight=float(config_get(cfg, "planning.uncertainty_weight", 0.05)),
    )


DUC_METHODS = {
    "duc_wm",
    "duc_random",
    "duc_no_llm",
    "duc_wrong_prior",
    "duc_no_unknown",
    "duc_no_reg",
    "duc_no_rollout",
}

BASELINE_METHODS = {"mlp", "pets", "cadm"}
ALL_METHODS = DUC_METHODS | BASELINE_METHODS


def duc_trainer_config(cfg: DictConfig, seed: int, method: str) -> DUCTrainerConfig:
    residual_weight = float(cfg.duc.get("residual_weight", 0.0))
    rollout_weight = float(cfg.duc.rollout_weight)
    orth_weight = float(cfg.duc.orth_weight)
    sparse_weight = float(cfg.duc.sparse_weight)
    unknown_weight = float(cfg.duc.get("unknown_weight", 0.0))
    trust_region_weight = float(config_get(cfg, "duc.trust_region_weight", 0.0))
    prior_beta_weight = float(config_get(cfg, "duc.prior_beta_weight", 0.0))
    residual_warmup_fraction = float(config_get(cfg, "duc.residual_warmup_fraction", 0.0))
    if method == "duc_no_reg":
        residual_weight = 0.0
        orth_weight = 0.0
        sparse_weight = 0.0
        unknown_weight = 0.0
        trust_region_weight = 0.0
        prior_beta_weight = 0.0
        residual_warmup_fraction = 0.0
    if method == "duc_no_rollout":
        rollout_weight = 0.0
    return DUCTrainerConfig(
        epochs=int(cfg.epochs),
        batch_size=int(cfg.batch_size),
        learning_rate=float(cfg.learning_rate),
        history_length=int(cfg.duc.history_length),
        beta_kl=float(cfg.duc.beta_kl),
        context_weight=float(cfg.duc.context_weight),
        residual_weight=residual_weight,
        control_weight=float(cfg.duc.control_weight),
        rollout_weight=rollout_weight,
        rollout_horizon=int(cfg.duc.rollout_horizon),
        orth_weight=orth_weight,
        sparse_weight=sparse_weight,
        unknown_weight=unknown_weight,
        trust_region_weight=trust_region_weight,
        trust_region_delta_min=float(config_get(cfg, "duc.trust_region_delta_min", 0.25)),
        trust_region_delta_range=float(config_get(cfg, "duc.trust_region_delta_range", 1.25)),
        prior_beta_weight=prior_beta_weight,
        residual_warmup_fraction=residual_warmup_fraction,
        teacher_force_context=bool(cfg.duc.teacher_force_context),
        seed=seed,
        precision=str(config_get(cfg, "runtime.precision", "fp32")),
        preload_to_device=bool(config_get(cfg, "runtime.preload_to_device", False)),
    )


def config_get(cfg: DictConfig, dotted_key: str, default: Any) -> Any:
    current: Any = cfg
    for part in dotted_key.split("."):
        if not hasattr(current, "get"):
            return default
        current = current.get(part)
        if current is None:
            return default
    return current


def configure_runtime(cfg: DictConfig) -> None:
    runtime = cfg.get("runtime", {})
    if runtime is None:
        runtime = {}
    num_threads = int(runtime.get("num_threads", 0))
    if num_threads > 0:
        torch.set_num_threads(num_threads)
    interop_threads = int(runtime.get("interop_threads", 0))
    if interop_threads > 0:
        try:
            torch.set_num_interop_threads(interop_threads)
        except RuntimeError:
            pass
    matmul_precision = str(runtime.get("matmul_precision", "high"))
    if matmul_precision:
        torch.set_float32_matmul_precision(matmul_precision)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = bool(runtime.get("cudnn_benchmark", True))
        allow_tf32 = bool(runtime.get("allow_tf32", True))
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32


def maybe_compile_forward(model: torch.nn.Module, cfg: DictConfig) -> None:
    if not bool(config_get(cfg, "runtime.compile", False)):
        return
    if not hasattr(torch, "compile"):
        return
    mode = str(config_get(cfg, "runtime.compile_mode", "max-autotune"))
    try:
        model.forward = torch.compile(model.forward, mode=mode, fullgraph=False)  # type: ignore[method-assign]
    except Exception as exc:
        print(f"warning: torch.compile setup failed; continuing without compile: {exc}")


def resolve_variants(raw: Any, fallback: str) -> list[str]:
    if raw is None:
        return [fallback]
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, (list, tuple)):
        values = list(raw)
    else:
        values = OmegaConf.to_container(raw, resolve=True)
    if isinstance(values, str):
        return [values]
    if not isinstance(values, list) or not values:
        raise ValueError("variant list must be a non-empty list or string")
    return [str(item) for item in values]


def collect_mujoco_variants(
    problem: str,
    total_steps: int,
    seed: int,
    variants: list[str],
    num_contexts: int,
    cfg: DictConfig,
) -> MuJoCoTransitions:
    if len(variants) == 1:
        return collect_mujoco_extension_dataset(
            MuJoCoExtensionConfig(
                env_id=problem,
                num_steps=total_steps,
                seed=seed,
                variant=variants[0],
                num_contexts=num_contexts,
                action_policy=str(cfg.mujoco_extension.action_policy),
                action_smoothing=float(cfg.mujoco_extension.action_smoothing),
                parallel_workers=int(cfg.mujoco_extension.get("parallel_workers", 1)),
            )
        )

    step_counts = distribute_count(total_steps, len(variants))
    context_counts = distribute_count(max(num_contexts, len(variants)), len(variants))
    datasets = []
    for index, variant in enumerate(variants):
        datasets.append(
            collect_mujoco_extension_dataset(
                MuJoCoExtensionConfig(
                    env_id=problem,
                    num_steps=step_counts[index],
                    seed=seed + 1000 * index,
                    variant=variant,
                    num_contexts=max(1, context_counts[index]),
                    action_policy=str(cfg.mujoco_extension.action_policy),
                    action_smoothing=float(cfg.mujoco_extension.action_smoothing),
                    parallel_workers=int(cfg.mujoco_extension.get("parallel_workers", 1)),
                )
            )
        )
    return concatenate_mujoco_transitions(datasets)


def distribute_count(total: int, buckets: int) -> list[int]:
    if total <= 0:
        raise ValueError("total must be positive")
    if buckets <= 0:
        raise ValueError("buckets must be positive")
    base = total // buckets
    remainder = total % buckets
    return [base + (1 if index < remainder else 0) for index in range(buckets)]


def variant_label(variants: list[str]) -> str:
    return "|".join(variants)


def safe_label(label: str) -> str:
    safe = label.replace("+", "plus").replace("|", "__").replace("/", "_").replace(":", "_")
    safe = safe.replace(",", "_").replace(" ", "")
    return safe[:160]


if __name__ == "__main__":
    main()
