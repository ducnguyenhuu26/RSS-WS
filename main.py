from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from onelife.duc_wm import (
    DUCLLMPriorConfig,
    DUCTrainerConfig,
    DUCWorldModel,
    DUCWorldModelConfig,
    MuJoCoExtensionConfig,
    build_duc_mujoco_prior_prompt,
    collect_mujoco_extension_dataset,
    default_mujoco_templates,
    evaluate_duc_model,
    fit_duc_world_model,
    load_templates_from_json_file,
    prompt_payload,
    synthesize_templates_with_llm,
)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    seed = int(cfg.seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = resolve_torch_device(str(cfg.get("device", "auto")))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    problem = str(cfg.problem)
    output_path = build_output_path(cfg, problem)
    if bool(cfg.get("skip_existing", False)) and output_path.exists():
        print(f"skipping existing output {output_path}")
        return

    train_dataset = collect_mujoco_extension_dataset(
        MuJoCoExtensionConfig(
            env_id=problem,
            num_steps=int(cfg.train_steps),
            seed=seed,
            variant=str(cfg.mujoco_extension.variant),
            num_contexts=int(cfg.mujoco_extension.num_contexts),
            action_policy=str(cfg.mujoco_extension.action_policy),
            action_smoothing=float(cfg.mujoco_extension.action_smoothing),
        )
    )
    test_dataset = collect_mujoco_extension_dataset(
        MuJoCoExtensionConfig(
            env_id=problem,
            num_steps=int(cfg.test_steps),
            seed=seed + 10_000,
            variant=str(cfg.mujoco_extension.test_variant),
            num_contexts=int(cfg.mujoco_extension.test_num_contexts),
            action_policy=str(cfg.mujoco_extension.action_policy),
            action_smoothing=float(cfg.mujoco_extension.action_smoothing),
        )
    )

    prior_prompt = build_duc_mujoco_prior_prompt(
        env_id=problem,
        state_dim=train_dataset.state_dim,
        action_dim=train_dataset.action_dim,
    )
    prior_path = cfg.duc.llm_prior.get("json_path")
    llm_prior_status: dict[str, Any] = {
        "source": "fallback",
        "error": None,
        "raw_response": None,
    }
    if prior_path:
        templates = load_templates_from_json_file(
            Path(str(prior_path)),
            state_dim=train_dataset.state_dim,
            action_dim=train_dataset.action_dim,
        )
        llm_prior_status["source"] = f"json_path:{prior_path}"
    elif bool(cfg.duc.llm_prior.get("enabled", False)):
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
    model = DUCWorldModel(
        DUCWorldModelConfig(
            state_dim=train_dataset.state_dim,
            action_dim=train_dataset.action_dim,
            templates=templates,
            hidden_size=int(cfg.duc.hidden_size),
            hidden_layers=int(cfg.duc.hidden_layers),
            history_length=int(cfg.duc.history_length),
        )
    ).to(device)
    history = fit_duc_world_model(
        model=model,
        transitions=train_dataset,
        config=DUCTrainerConfig(
            epochs=int(cfg.epochs),
            batch_size=int(cfg.batch_size),
            learning_rate=float(cfg.learning_rate),
            history_length=int(cfg.duc.history_length),
            beta_kl=float(cfg.duc.beta_kl),
            context_weight=float(cfg.duc.context_weight),
            control_weight=float(cfg.duc.control_weight),
            rollout_weight=float(cfg.duc.rollout_weight),
            rollout_horizon=int(cfg.duc.rollout_horizon),
            orth_weight=float(cfg.duc.orth_weight),
            sparse_weight=float(cfg.duc.sparse_weight),
            teacher_force_context=bool(cfg.duc.teacher_force_context),
            seed=seed,
        ),
        device=device,
    )
    metrics = evaluate_duc_model(
        model=model,
        transitions=test_dataset,
        device=device,
        batch_size=int(cfg.eval_batch_size),
        history_length=int(cfg.duc.history_length),
    )
    payload = {
        "framework": "DUC-WM",
        "model": "duc_wm",
        "problem": problem,
        "seed": seed,
        "device": str(device),
        "train_steps": train_dataset.num_steps,
        "test_steps": test_dataset.num_steps,
        "variant": str(cfg.mujoco_extension.variant),
        "test_variant": str(cfg.mujoco_extension.test_variant),
        "context_names": list(train_dataset.context_names),
        "llm_prior_prompt": prompt_payload(prior_prompt),
        "llm_prior_status": llm_prior_status,
        "mechanisms": [
            {
                "name": template.name,
                "state_indices": list(template.state_indices),
                "action_indices": list(template.action_indices),
                "output_indices": list(template.output_indices),
                "scale": template.scale,
                "prior_std": template.prior_std,
                "description": template.description,
            }
            for template in templates
        ],
        "training": history,
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
    safe_problem = problem.replace("/", "_").replace(":", "_")
    return output_dir / f"duc_wm_{safe_problem}_seed{int(cfg.seed)}.json"


if __name__ == "__main__":
    main()
