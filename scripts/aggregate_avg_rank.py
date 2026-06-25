from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate SimFutures-LP benchmark JSON files into per-environment ranks "
            "and method-level average rank. Pass multiple metrics for a "
            "composite AvgRank over env-metric pairs."
        )
    )
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--metric", default="score.planner_return_mean")
    parser.add_argument(
        "--metrics",
        default=None,
        help=(
            "Comma-separated metrics for composite AvgRank. If omitted, "
            "--metric is used for backward compatibility."
        ),
    )
    parser.add_argument("--lower-is-better", action="store_true")
    parser.add_argument("--group-by", default="problem,test_variant")
    parser.add_argument("--method-key", default="model")
    parser.add_argument("--precision", type=int, default=4)
    parser.add_argument("--csv-out", default=None)
    args = parser.parse_args()

    metric_keys = (
        [item.strip() for item in args.metrics.split(",") if item.strip()]
        if args.metrics
        else [args.metric]
    )
    rows = load_rows(
        paths=expand_input_paths(args.files),
        group_keys=[item.strip() for item in args.group_by.split(",") if item.strip()],
        method_key=args.method_key,
        metric_keys=metric_keys,
    )
    if not rows:
        raise SystemExit("no rows found")

    grouped: dict[tuple[Any, ...], dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for row in rows:
        grouped[row["group"]][row["metric"]][row["method"]].append(row["value"])

    fmt = f"{{:.{args.precision}f}}"
    method_ranks: dict[str, list[float]] = defaultdict(list)
    method_metric_scores: dict[str, list[float]] = defaultdict(list)
    csv_rows: list[dict[str, Any]] = []

    print("metrics:", ",".join(metric_keys))
    print("group_by:", args.group_by)
    print()

    for group, metric_map in sorted(grouped.items()):
        group_label = " | ".join(str(item) for item in group)
        for metric in metric_keys:
            methods = metric_map.get(metric, {})
            if not methods:
                continue
            summaries = []
            for method, values in methods.items():
                avg = mean(values)
                spread = stdev(values) if len(values) > 1 else 0.0
                summaries.append(
                    {
                        "method": method,
                        "mean": avg,
                        "std": spread,
                        "n": len(values),
                    }
                )
            summaries.sort(key=lambda item: item["mean"], reverse=not args.lower_is_better)

            print(f"{group_label} | metric={metric}")
            for rank, item in enumerate(summaries, start=1):
                method = str(item["method"])
                method_ranks[method].append(float(rank))
                method_metric_scores[method].append(float(item["mean"]))
                print(
                    f"  {rank}. {method}: "
                    f"{fmt.format(item['mean'])} +/- {fmt.format(item['std'])} "
                    f"(n={item['n']})"
                )
                csv_rows.append(
                    {
                        "group": group_label,
                        "metric": metric,
                        "method": method,
                        "rank": rank,
                        "mean": item["mean"],
                        "std": item["std"],
                        "n": item["n"],
                    }
                )
            print()

    avg_rank_rows = []
    for method, ranks in method_ranks.items():
        rank_mean = mean(ranks)
        rank_std = stdev(ranks) if len(ranks) > 1 else 0.0
        score_values = method_metric_scores[method]
        score_mean = mean(score_values)
        score_std = stdev(score_values) if len(score_values) > 1 else 0.0
        avg_rank_rows.append(
            {
                "method": method,
                "avg_rank": rank_mean,
                "std_rank": rank_std,
                "rank_items": len(ranks),
                "mean_metric": score_mean,
                "std_metric": score_std,
            }
        )
    avg_rank_rows.sort(key=lambda item: item["avg_rank"])

    print("avg_rank:")
    for rank, item in enumerate(avg_rank_rows, start=1):
        print(
            f"  {rank}. {item['method']}: "
            f"avg_rank={fmt.format(item['avg_rank'])} "
            f"+/- {fmt.format(item['std_rank'])} "
            f"rank_items={item['rank_items']} "
            f"mean_metric={fmt.format(item['mean_metric'])} "
            f"+/- {fmt.format(item['std_metric'])}"
        )

    if args.csv_out:
        write_csv(Path(args.csv_out), csv_rows, avg_rank_rows)


def load_rows(
    paths: list[Path],
    group_keys: list[str],
    method_key: str,
    metric_keys: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        try:
            method = str(get_nested(payload, method_key))
            group = tuple(get_nested(payload, key) for key in group_keys)
        except KeyError:
            continue
        for metric_key in metric_keys:
            try:
                value = float(get_nested(payload, metric_key))
            except KeyError:
                continue
            rows.append(
                {
                    "path": path,
                    "method": method,
                    "group": group,
                    "metric": metric_key,
                    "value": value,
                }
            )
    return rows


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


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    avg_rank_rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["section", "group", "metric", "method", "rank", "mean", "std", "n"])
        for row in rows:
            writer.writerow(
                [
                    "env_metric",
                    row["group"],
                    row["metric"],
                    row["method"],
                    row["rank"],
                    row["mean"],
                    row["std"],
                    row["n"],
                ]
            )
        writer.writerow([])
        writer.writerow(
            [
                "section",
                "method",
                "avg_rank",
                "std_rank",
                "rank_items",
                "mean_metric",
                "std_metric",
            ]
        )
        for row in avg_rank_rows:
            writer.writerow(
                [
                    "avg_rank",
                    row["method"],
                    row["avg_rank"],
                    row["std_rank"],
                    row["rank_items"],
                    row["mean_metric"],
                    row["std_metric"],
                ]
            )


if __name__ == "__main__":
    main()
