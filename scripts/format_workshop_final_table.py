from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


METRICS = (
    ("score.r2_at_1", "R2@1"),
    ("score.r2_at_10", "R2@10"),
    ("score.r2_at_25", "R2@25"),
    ("score.planner_return_mean", "Reward"),
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Format the final workshop table: R2@1, R2@10, R2@25, Reward, and "
            "composite AvgRank over environment-metric pairs."
        )
    )
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--precision", type=int, default=3)
    parser.add_argument("--csv-out", default=None)
    args = parser.parse_args()

    paths = expand_input_paths(args.files)
    env_metric_values = load_env_metric_values(paths)
    if not env_metric_values:
        raise SystemExit("no metric rows found")

    env_metric_means = summarize_env_metric_values(env_metric_values)
    method_ranks = compute_method_ranks(env_metric_means)
    method_metric_values = collect_method_metric_env_means(env_metric_means)
    rows = build_table_rows(method_metric_values, method_ranks)

    print_markdown_table(rows, precision=args.precision)
    if args.csv_out:
        write_csv(Path(args.csv_out), rows)


def load_env_metric_values(
    paths: list[Path],
) -> dict[tuple[str, str, str], list[float]]:
    values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        try:
            method = str(get_nested(payload, "model"))
            problem = str(get_nested(payload, "problem"))
            test_variant = str(get_nested(payload, "test_variant"))
        except KeyError:
            continue
        env = f"{problem}:{test_variant}"
        for metric_key, _ in METRICS:
            try:
                value = float(get_nested(payload, metric_key))
            except KeyError:
                continue
            values[(env, method, metric_key)].append(value)
    return values


def summarize_env_metric_values(
    values: dict[tuple[str, str, str], list[float]],
) -> dict[tuple[str, str, str], dict[str, float]]:
    summary = {}
    for key, items in values.items():
        summary[key] = {
            "mean": mean(items),
            "std": stdev(items) if len(items) > 1 else 0.0,
            "n": float(len(items)),
        }
    return summary


def compute_method_ranks(
    env_metric_means: dict[tuple[str, str, str], dict[str, float]],
) -> dict[str, list[float]]:
    by_env_metric: dict[tuple[str, str], list[tuple[str, float]]] = defaultdict(list)
    for (env, method, metric), summary in env_metric_means.items():
        by_env_metric[(env, metric)].append((method, summary["mean"]))

    method_ranks: dict[str, list[float]] = defaultdict(list)
    for (_env, _metric), items in by_env_metric.items():
        items.sort(key=lambda item: item[1], reverse=True)
        for rank, (method, _value) in enumerate(items, start=1):
            method_ranks[method].append(float(rank))
    return method_ranks


def collect_method_metric_env_means(
    env_metric_means: dict[tuple[str, str, str], dict[str, float]],
) -> dict[str, dict[str, list[float]]]:
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (_env, method, metric), summary in env_metric_means.items():
        values[method][metric].append(summary["mean"])
    return values


def build_table_rows(
    method_metric_values: dict[str, dict[str, list[float]]],
    method_ranks: dict[str, list[float]],
) -> list[dict[str, Any]]:
    rows = []
    for method in sorted(method_metric_values):
        row: dict[str, Any] = {"method": method}
        for metric, label in METRICS:
            values = method_metric_values[method].get(metric, [])
            if values:
                row[label] = {
                    "mean": mean(values),
                    "std": stdev(values) if len(values) > 1 else 0.0,
                    "n_env": len(values),
                }
        ranks = method_ranks.get(method, [])
        if ranks:
            row["AvgRank"] = {
                "mean": mean(ranks),
                "std": stdev(ranks) if len(ranks) > 1 else 0.0,
                "n_items": len(ranks),
            }
        rows.append(row)
    rows.sort(key=lambda item: item.get("AvgRank", {}).get("mean", float("inf")))
    return rows


def print_markdown_table(rows: list[dict[str, Any]], precision: int) -> None:
    headers = ["Method", "R2@1", "R2@10", "R2@25", "Reward", "AvgRank"]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        print(
            "| "
            + " | ".join(
                [
                    str(row["method"]),
                    format_cell(row.get("R2@1"), precision),
                    format_cell(row.get("R2@10"), precision),
                    format_cell(row.get("R2@25"), precision),
                    format_cell(row.get("Reward"), precision),
                    format_rank(row.get("AvgRank"), precision),
                ]
            )
            + " |"
        )
    print()
    print(
        "Note: metric cells are mean +/- std over environment-level seed means. "
        "AvgRank is averaged over each environment x metric rank; lower is better."
    )


def format_cell(value: dict[str, float] | None, precision: int) -> str:
    if not value:
        return "-"
    return f"{value['mean']:.{precision}f} +/- {value['std']:.{precision}f}"


def format_rank(value: dict[str, float] | None, precision: int) -> str:
    if not value:
        return "-"
    return f"{value['mean']:.{precision}f}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "r2_at_1_mean",
                "r2_at_1_std",
                "r2_at_10_mean",
                "r2_at_10_std",
                "r2_at_25_mean",
                "r2_at_25_std",
                "reward_mean",
                "reward_std",
                "avg_rank",
                "avg_rank_std",
                "rank_items",
            ]
        )
        for row in rows:
            r2_1 = row.get("R2@1", {})
            r2_10 = row.get("R2@10", {})
            r2_25 = row.get("R2@25", {})
            reward = row.get("Reward", {})
            rank = row.get("AvgRank", {})
            writer.writerow(
                [
                    row["method"],
                    r2_1.get("mean"),
                    r2_1.get("std"),
                    r2_10.get("mean"),
                    r2_10.get("std"),
                    r2_25.get("mean"),
                    r2_25.get("std"),
                    reward.get("mean"),
                    reward.get("std"),
                    rank.get("mean"),
                    rank.get("std"),
                    rank.get("n_items"),
                ]
            )


def get_nested(payload: dict[str, Any], dotted_key: str) -> Any:
    current: Any = payload
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"missing {dotted_key}")
        current = current[part]
    return current


def expand_input_paths(paths: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        raw = str(path)
        if any(char in raw for char in "*?["):
            expanded.extend(Path(match) for match in sorted(glob.glob(raw)))
        else:
            expanded.append(path)
    return expanded


if __name__ == "__main__":
    main()
