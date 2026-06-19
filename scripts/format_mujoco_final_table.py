from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


ENV_ORDER = [
    "Swimmer-v5",
    "InvertedDoublePendulum-v5",
    "Reacher-v5",
    "Hopper-v5",
    "Walker2d-v5",
    "HalfCheetah-v5",
]

EXCLUDED_ENVS = {
    "InvertedPendulum-v5",
    "Ant-v5",
    "Pusher-v5",
}

MODEL_ORDER = [
    "onelife",
    "pets_ensemble",
    "dreamer_v3",
    "neural",
    "program_only",
    "symbolic_neural",
    "neural_mlp",
    "answer",
]

MODEL_LABELS = {
    "answer": "ANSWER",
    "onelife": "OneLife",
    "pets_ensemble": "PETS",
    "dreamer_v3": "DreamerV3",
    "neural": "ODE-only",
    "neural_mlp": "MLP-only",
    "program_only": "LLM-only",
    "symbolic_neural": "Lib+ODE",
}

R2_AT_1_KEY = "score.r2_at_1_delta_uniform"
R2_AT_10_KEY = "score.r2_at_10_delta_uniform"
CEM_REWARD_KEY = "reward.cem_mpc_return_mean"
CEM_PEC_REWARD_KEY = "reward.cem_pec_mpc_return_mean"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Format final MuJoCo result tables from per-seed JSON outputs."
    )
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--precision", type=int, default=3)
    parser.add_argument("--no-std", action="store_true")
    args = parser.parse_args()

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in expand_input_paths(args.files):
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        env = payload.get("problem") or payload.get("env_id")
        model = payload.get("model")
        if env is None or model is None:
            continue
        if env in EXCLUDED_ENVS:
            continue
        grouped[(str(env), str(model))].append(payload)

    for env in _ordered_envs(grouped):
        print(f"### {env}")
        reward_header, reward_mode = _reward_header_and_mode(
            env,
            grouped,
            MODEL_ORDER,
        )
        print(f"| Model | R2@1 | R2@10 | {reward_header} |")
        print("|---|---:|---:|---:|")
        _print_model_rows(
            env,
            grouped,
            MODEL_ORDER,
            args.precision,
            args.no_std,
            reward_mode,
        )
        print()


def _print_model_rows(
    env: str,
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    model_order: list[str],
    precision: int,
    no_std: bool,
    reward_mode: str,
) -> None:
    for model in model_order:
        rows = grouped.get((env, model), [])
        if not rows:
            continue
        r2_at_1 = _format_metric_with_fallback(
            rows,
            R2_AT_1_KEY,
            "score.one_step_delta_r2_uniform",
            precision,
            no_std,
        )
        r2_at_10 = _format_metric(rows, R2_AT_10_KEY, precision, no_std)
        cem_reward = _format_metric(rows, CEM_REWARD_KEY, precision, no_std)
        cem_pec_reward = _format_metric(rows, CEM_PEC_REWARD_KEY, precision, no_std)
        reward = _format_reward_cell(cem_reward, cem_pec_reward, reward_mode)
        label = MODEL_LABELS.get(model, model)
        print(f"| {label} | {r2_at_1} | {r2_at_10} | {reward} |")


def _reward_header_and_mode(
    env: str,
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    model_order: list[str],
) -> tuple[str, str]:
    rows = [
        row
        for model in model_order
        for row in grouped.get((env, model), [])
    ]
    has_cem = any(_has_nested(row, CEM_REWARD_KEY) for row in rows)
    has_pec = any(_has_nested(row, CEM_PEC_REWARD_KEY) for row in rows)
    if has_cem and has_pec:
        return "Reward (CEM-MPC / PEC-CEM-MPC)", "both"
    if has_cem:
        return "Reward (CEM-MPC)", "cem"
    if has_pec:
        return "Reward (PEC-CEM-MPC)", "pec"
    return "Reward", "none"


def _format_reward_cell(
    cem_reward: str,
    cem_pec_reward: str,
    reward_mode: str,
) -> str:
    if reward_mode == "both":
        return f"{cem_reward} / {cem_pec_reward}"
    if reward_mode == "cem":
        return cem_reward
    if reward_mode == "pec":
        return cem_pec_reward
    return "n/a"


def _ordered_envs(grouped: dict[tuple[str, str], list[dict[str, Any]]]) -> list[str]:
    available = {env for env, _model in grouped}
    ordered = [env for env in ENV_ORDER if env in available]
    ordered.extend(sorted(available - set(ordered)))
    return ordered


def expand_input_paths(paths: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        raw = str(path)
        if any(char in raw for char in "*?["):
            matches = [Path(match) for match in sorted(glob.glob(raw))]
            expanded.extend(matches or [path])
        else:
            expanded.append(path)
    return expanded


def _format_metric(
    rows: list[dict[str, Any]],
    key: str,
    precision: int,
    no_std: bool,
) -> str:
    values = [float(get_nested(row, key)) for row in rows if _has_nested(row, key)]
    if not values:
        return "n/a"
    avg = mean(values)
    fmt = f"{{:.{precision}f}}"
    if no_std or len(values) == 1:
        return fmt.format(avg)
    return f"{fmt.format(avg)} +/- {fmt.format(stdev(values))}"


def _format_metric_with_fallback(
    rows: list[dict[str, Any]],
    key: str,
    fallback_key: str,
    precision: int,
    no_std: bool,
) -> str:
    value = _format_metric(rows, key, precision, no_std)
    if value != "n/a":
        return value
    return _format_metric(rows, fallback_key, precision, no_std)


def _has_nested(payload: dict[str, Any], dotted_key: str) -> bool:
    current: Any = payload
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return False
        current = current[part]
    return True


def get_nested(payload: dict[str, Any], dotted_key: str) -> Any:
    current: Any = payload
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"metric '{dotted_key}' not found; missing '{part}'")
        current = current[part]
    return current


if __name__ == "__main__":
    main()
