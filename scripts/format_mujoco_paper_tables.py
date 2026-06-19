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
    "program_only": "LLM-only",
    "symbolic_neural": "Lib+ODE",
    "neural_mlp": "MLP-only",
}

GROUP_BREAK_BEFORE = {
    "neural",
    "answer",
}

R2_AT_1_KEY = "score.r2_at_1_delta_uniform"
R2_AT_1_FALLBACK_KEY = "score.one_step_delta_r2_uniform"
R2_AT_10_KEY = "score.r2_at_10_delta_uniform"
CEM_REWARD_KEY = "reward.cem_mpc_return_mean"
CEM_PEC_REWARD_KEY = "reward.cem_pec_mpc_return_mean"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Format compact paper tables for MuJoCo results."
    )
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--precision", type=int, default=3)
    parser.add_argument("--no-std", action="store_true")
    parser.add_argument(
        "--format",
        choices=("markdown", "latex"),
        default="markdown",
        help="markdown for terminal preview, latex for paper tables",
    )
    parser.add_argument(
        "--layout",
        choices=("combined", "split"),
        default="combined",
        help=(
            "combined prints one main table with R2@1, R2@10, and reward "
            "under each environment; split prints separate prediction and "
            "reward tables"
        ),
    )
    args = parser.parse_args()

    grouped = _load_grouped(args.files)
    envs = _ordered_envs(grouped)
    models = _ordered_models(grouped)

    if args.layout == "combined":
        if args.format == "latex":
            _print_latex_combined_table(grouped, envs, models, args.precision, args.no_std)
        else:
            _print_markdown_combined_table(
                grouped,
                envs,
                models,
                args.precision,
                args.no_std,
            )
    elif args.format == "latex":
        _print_latex_r2_table(grouped, envs, models, args.precision, args.no_std)
        print()
        _print_latex_reward_table(grouped, envs, models, args.precision, args.no_std)
    else:
        _print_markdown_r2_table(grouped, envs, models, args.precision, args.no_std)
        print()
        _print_markdown_reward_table(grouped, envs, models, args.precision, args.no_std)


def _load_grouped(
    files: list[Path],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in _expand_input_paths(files):
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        env = payload.get("problem") or payload.get("env_id")
        model = payload.get("model")
        if env is None or model is None:
            continue
        if env in EXCLUDED_ENVS:
            continue
        grouped[(str(env), str(model))].append(payload)
    return grouped


def _print_markdown_combined_table(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    envs: list[str],
    models: list[str],
    precision: int,
    no_std: bool,
) -> None:
    reward_name, reward_key = _reward_metric_choice(grouped)
    avg_ranks = _compute_average_ranks(grouped, envs, models, reward_key)
    print("### Main Table. Prediction and planning")
    header_top = ["Variant"]
    header_sub = [""]
    aligns = ["---"]
    for env in envs:
        header_top.extend([_short_env(env), "", ""])
        header_sub.extend(["R2@1", "R2@10", "Reward"])
        aligns.extend(["---:", "---:", "---:"])
    header_top.append("Avg. Rank")
    header_sub.append("")
    aligns.append("---:")
    print("| " + " | ".join(header_top) + " |")
    print("| " + " | ".join(header_sub) + " |")
    print("| " + " | ".join(aligns) + " |")
    for model in models:
        cells = [_model_label(model)]
        for env in envs:
            rows = grouped.get((env, model), [])
            cells.append(
                _format_metric_with_fallback(
                    rows,
                    R2_AT_1_KEY,
                    R2_AT_1_FALLBACK_KEY,
                    precision,
                    no_std,
                )
            )
            cells.append(_format_metric(rows, R2_AT_10_KEY, precision, no_std))
            cells.append(_format_metric(rows, reward_key, precision, no_std))
        cells.append(_format_average_rank(avg_ranks, model, precision))
        print("| " + " | ".join(cells) + " |")
    print()
    print(f"Reward metric: {reward_name}. Avg. Rank is lower-is-better.")


def _print_latex_combined_table(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    envs: list[str],
    models: list[str],
    precision: int,
    no_std: bool,
) -> None:
    reward_name, reward_key = _reward_metric_choice(grouped)
    avg_ranks = _compute_average_ranks(grouped, envs, models, reward_key)
    columns = "@{}l" + ("rrr" * len(envs)) + "r@{}"
    print(r"\begin{table*}[t]")
    print(r"\centering")
    print(
        rf"\caption{{MuJoCo-v5 prediction and planning results averaged over "
        rf"three seeds. Reward uses {_latex(reward_name)}. "
        rf"Avg. Rank is lower-is-better.}}"
    )
    print(r"\label{tab:mujoco-main}")
    print(r"\scriptsize")
    print(r"\setlength{\tabcolsep}{2.2pt}")
    print(r"\resizebox{\textwidth}{!}{%")
    print(rf"\begin{{tabular}}{{{columns}}}")
    print(r"\toprule")
    top = ["Variant"] + [
        rf"\multicolumn{{3}}{{c}}{{{_latex(_short_env(env))}}}"
        for env in envs
    ]
    top.append(r"Avg. Rank")
    print(" & ".join(top) + r" \\")
    cmidrules = [
        rf"\cmidrule(lr){{{2 + 3 * index}-{4 + 3 * index}}}"
        for index, _env in enumerate(envs)
    ]
    print(" ".join(cmidrules))
    sub = [""] + [
        item
        for _env in envs
        for item in (r"$R^2@1\uparrow$", r"$R^2@10\uparrow$", r"Reward$\uparrow$")
    ]
    sub.append(r"$\downarrow$")
    print(" & ".join(sub) + r" \\")
    print(r"\midrule")
    for model in models:
        if model in GROUP_BREAK_BEFORE:
            print(r"\midrule")
        cells = [_latex(_model_label(model))]
        for env in envs:
            rows = grouped.get((env, model), [])
            cells.append(
                _format_metric_with_fallback(
                    rows,
                    R2_AT_1_KEY,
                    R2_AT_1_FALLBACK_KEY,
                    precision,
                    no_std,
                )
            )
            cells.append(_format_metric(rows, R2_AT_10_KEY, precision, no_std))
            cells.append(_format_metric(rows, reward_key, precision, no_std))
        cells.append(_format_average_rank(avg_ranks, model, precision))
        print(" & ".join(cells) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"}")
    print(r"\end{table*}")


def _print_markdown_r2_table(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    envs: list[str],
    models: list[str],
    precision: int,
    no_std: bool,
) -> None:
    print("### Table 1. Prediction accuracy")
    header_top = ["Model"]
    header_sub = [""]
    aligns = ["---"]
    for env in envs:
        header_top.extend([_short_env(env), ""])
        header_sub.extend(["R2@1", "R2@10"])
        aligns.extend(["---:", "---:"])
    print("| " + " | ".join(header_top) + " |")
    print("| " + " | ".join(header_sub) + " |")
    print("| " + " | ".join(aligns) + " |")
    for model in models:
        cells = [_model_label(model)]
        for env in envs:
            rows = grouped.get((env, model), [])
            cells.append(
                _format_metric_with_fallback(
                    rows,
                    R2_AT_1_KEY,
                    R2_AT_1_FALLBACK_KEY,
                    precision,
                    no_std,
                )
            )
            cells.append(_format_metric(rows, R2_AT_10_KEY, precision, no_std))
        print("| " + " | ".join(cells) + " |")


def _print_markdown_reward_table(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    envs: list[str],
    models: list[str],
    precision: int,
    no_std: bool,
) -> None:
    print("### Table 2. Planning return")
    reward_name, reward_key = _reward_metric_choice(grouped)
    header = ["Model", *[_short_env(env) for env in envs]]
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join(["---", *(["---:"] * len(envs))]) + " |")
    for model in models:
        cells = [_model_label(model)]
        for env in envs:
            cells.append(
                _format_metric(grouped.get((env, model), []), reward_key, precision, no_std)
            )
        print("| " + " | ".join(cells) + " |")
    print()
    print(f"Reward metric: {reward_name}.")


def _print_latex_r2_table(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    envs: list[str],
    models: list[str],
    precision: int,
    no_std: bool,
) -> None:
    columns = "l" + ("rr" * len(envs))
    print(r"\begin{table*}[t]")
    print(r"\centering")
    print(r"\caption{Prediction accuracy on MuJoCo-v5.}")
    print(r"\label{tab:mujoco-prediction}")
    print(rf"\begin{{tabular}}{{{columns}}}")
    print(r"\toprule")
    top = ["Model"] + [rf"\multicolumn{{2}}{{c}}{{{_latex(_short_env(env))}}}" for env in envs]
    print(" & ".join(top) + r" \\")
    sub = [""] + [item for _env in envs for item in (r"$R^2@1$", r"$R^2@10$")]
    print(" & ".join(sub) + r" \\")
    print(r"\midrule")
    for model in models:
        cells = [_latex(_model_label(model))]
        for env in envs:
            rows = grouped.get((env, model), [])
            cells.append(
                _format_metric_with_fallback(
                    rows,
                    R2_AT_1_KEY,
                    R2_AT_1_FALLBACK_KEY,
                    precision,
                    no_std,
                )
            )
            cells.append(_format_metric(rows, R2_AT_10_KEY, precision, no_std))
        print(" & ".join(cells) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table*}")


def _print_latex_reward_table(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    envs: list[str],
    models: list[str],
    precision: int,
    no_std: bool,
) -> None:
    reward_name, reward_key = _reward_metric_choice(grouped)
    columns = "l" + ("r" * len(envs))
    print(r"\begin{table*}[t]")
    print(r"\centering")
    print(rf"\caption{{Planning return using {_latex(reward_name)}.}}")
    print(r"\label{tab:mujoco-planning}")
    print(rf"\begin{{tabular}}{{{columns}}}")
    print(r"\toprule")
    print(" & ".join(["Model", *[_latex(_short_env(env)) for env in envs]]) + r" \\")
    print(r"\midrule")
    for model in models:
        cells = [_latex(_model_label(model))]
        for env in envs:
            cells.append(
                _format_metric(grouped.get((env, model), []), reward_key, precision, no_std)
            )
        print(" & ".join(cells) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table*}")


def _reward_metric_choice(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
) -> tuple[str, str]:
    rows = [row for payloads in grouped.values() for row in payloads]
    has_pec = any(_has_nested(row, CEM_PEC_REWARD_KEY) for row in rows)
    has_cem = any(_has_nested(row, CEM_REWARD_KEY) for row in rows)
    if has_pec:
        return "PEC-CEM-MPC", CEM_PEC_REWARD_KEY
    if has_cem:
        return "CEM-MPC", CEM_REWARD_KEY
    return "Reward", CEM_PEC_REWARD_KEY


def _ordered_envs(grouped: dict[tuple[str, str], list[dict[str, Any]]]) -> list[str]:
    available = {env for env, _model in grouped}
    ordered = [env for env in ENV_ORDER if env in available]
    ordered.extend(sorted(available - set(ordered)))
    return ordered


def _ordered_models(grouped: dict[tuple[str, str], list[dict[str, Any]]]) -> list[str]:
    available = {model for _env, model in grouped}
    ordered = [model for model in MODEL_ORDER if model in available]
    ordered.extend(sorted(available - set(ordered)))
    return ordered


def _expand_input_paths(paths: list[Path]) -> list[Path]:
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
    values = [float(_get_nested(row, key)) for row in rows if _has_nested(row, key)]
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


def _compute_average_ranks(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    envs: list[str],
    models: list[str],
    reward_key: str,
) -> tuple[dict[str, float], dict[str, int], int]:
    rank_sums = {model: 0.0 for model in models}
    rank_counts = {model: 0 for model in models}
    metric_specs = [
        (R2_AT_1_KEY, R2_AT_1_FALLBACK_KEY),
        (R2_AT_10_KEY, None),
        (reward_key, None),
    ]
    total_ranked_metrics = 0
    for env in envs:
        for key, fallback_key in metric_specs:
            values: list[tuple[str, float]] = []
            for model in models:
                value = _metric_mean_with_fallback(
                    grouped.get((env, model), []),
                    key,
                    fallback_key,
                )
                if value is not None:
                    values.append((model, value))
            if not values:
                continue
            total_ranked_metrics += 1
            values.sort(key=lambda item: item[1], reverse=True)
            index = 0
            while index < len(values):
                end = index + 1
                while end < len(values) and values[end][1] == values[index][1]:
                    end += 1
                rank = mean(range(index + 1, end + 1))
                for tied_index in range(index, end):
                    model = values[tied_index][0]
                    rank_sums[model] += rank
                    rank_counts[model] += 1
                index = end
    return rank_sums, rank_counts, total_ranked_metrics


def _format_average_rank(
    ranks: tuple[dict[str, float], dict[str, int], int],
    model: str,
    precision: int,
) -> str:
    rank_sums, rank_counts, total_ranked_metrics = ranks
    if total_ranked_metrics == 0 or rank_counts.get(model, 0) != total_ranked_metrics:
        return "n/a"
    return f"{rank_sums[model] / rank_counts[model]:.{precision}f}"


def _metric_mean_with_fallback(
    rows: list[dict[str, Any]],
    key: str,
    fallback_key: str | None,
) -> float | None:
    value = _metric_mean(rows, key)
    if value is None and fallback_key is not None:
        value = _metric_mean(rows, fallback_key)
    return value


def _metric_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(_get_nested(row, key)) for row in rows if _has_nested(row, key)]
    if not values:
        return None
    return mean(values)


def _has_nested(payload: dict[str, Any], dotted_key: str) -> bool:
    current: Any = payload
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return False
        current = current[part]
    return True


def _get_nested(payload: dict[str, Any], dotted_key: str) -> Any:
    current: Any = payload
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"metric '{dotted_key}' not found; missing '{part}'")
        current = current[part]
    return current


def _short_env(env: str) -> str:
    labels = {
        "Swimmer-v5": "Swimmer",
        "InvertedDoublePendulum-v5": "Inv.DPend.",
        "Reacher-v5": "Reacher",
        "Hopper-v5": "Hopper",
        "Walker2d-v5": "Walker2d",
        "HalfCheetah-v5": "HalfCheetah",
    }
    return labels.get(env, env.removesuffix("-v5"))


def _model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def _latex(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
    )


if __name__ == "__main__":
    main()
