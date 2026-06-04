from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


ENV_ORDER = [
    "InvertedPendulum-v5",
    "InvertedDoublePendulum-v5",
    "Reacher-v5",
    "Swimmer-v5",
    "Hopper-v5",
    "Walker2d-v5",
    "HalfCheetah-v5",
]

MODEL_ORDER = [
    "onelife",
    "ours_new",
    "ours_gated_island",
    "ours_gated",
    "ours",
    "program_only",
    "neural",
    "symbolic",
    "symbolic_neural",
]

MODEL_LABELS = {
    "onelife": "Adaptive OneLife",
    "ours_new": "Ours New: niche-island gated LLM + neural",
    "ours_gated_island": "Ours: niche-island gated LLM + neural",
    "ours_gated": "Ours: gated LLM symbolic + neural",
    "ours": "Ours: LLM symbolic + neural",
    "program_only": "Ours: LLM symbolic only",
    "neural": "Ours: neural only",
    "symbolic": "Ours: standard symbolic only",
    "symbolic_neural": "Ours: standard symbolic + neural",
}

SCORE_KEY = "score.one_step_delta_r2_uniform"
RANDOM_REWARD_KEY = "reward.random_mpc_return_mean"
CEM_REWARD_KEY = "reward.cem_mpc_return_mean"


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
        payload = json.loads(path.read_text(encoding="utf-8"))
        env = payload.get("problem") or payload.get("env_id")
        model = payload.get("model")
        if env is None or model is None:
            continue
        if env in {"Ant-v5", "Pusher-v5"}:
            continue
        grouped[(str(env), str(model))].append(payload)

    for env in _ordered_envs(grouped):
        print(f"### {env}")
        print("| Model | Score (R2) | Reward (Random MPC / CEM-MPC) |")
        print("|---|---:|---:|")
        for model in MODEL_ORDER:
            rows = grouped.get((env, model), [])
            if not rows:
                continue
            score = _format_metric(rows, SCORE_KEY, args.precision, args.no_std)
            random_reward = _format_metric(
                rows, RANDOM_REWARD_KEY, args.precision, args.no_std
            )
            cem_reward = _format_metric(rows, CEM_REWARD_KEY, args.precision, args.no_std)
            label = MODEL_LABELS.get(model, model)
            print(f"| {label} | {score} | {random_reward} / {cem_reward} |")
        print()


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
