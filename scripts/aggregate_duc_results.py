from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


DEFAULT_METRICS = (
    "score.r2_at_1",
    "score.r2_at_10",
    "score.duc_r2_at_10",
    "score.nll",
    "score.attribution_recall_at_2",
    "score.strength_spearman",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate DUC-WM benchmark JSON outputs by method/problem/split."
    )
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument(
        "--group-by",
        default="model,problem,test_variant",
        help="comma-separated payload keys",
    )
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help="comma-separated dotted metric keys",
    )
    parser.add_argument("--rank-metric", default="score.duc_r2_at_10")
    parser.add_argument("--rank-lower-is-better", action="store_true")
    parser.add_argument("--precision", type=int, default=4)
    args = parser.parse_args()

    group_keys = [item.strip() for item in args.group_by.split(",") if item.strip()]
    metric_keys = [item.strip() for item in args.metrics.split(",") if item.strip()]
    rows = load_rows(expand_input_paths(args.files), group_keys, metric_keys, args.rank_metric)
    if not rows:
        raise SystemExit("no rows found")

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["group"]].append(row)

    fmt = f"{{:.{args.precision}f}}"
    print("group_by:", ",".join(group_keys))
    print("metrics:", ",".join(metric_keys))
    print()

    summaries = []
    for group, items in sorted(grouped.items()):
        label = " | ".join(f"{key}={value}" for key, value in zip(group_keys, group, strict=True))
        print(label)
        summary: dict[str, Any] = {"group": group, "label": label}
        for metric in metric_keys:
            values = [item["metrics"][metric] for item in items if metric in item["metrics"]]
            if not values:
                continue
            avg = mean(values)
            spread = stdev(values) if len(values) > 1 else 0.0
            summary[metric] = avg
            print(f"  {metric}: {fmt.format(avg)} +/- {fmt.format(spread)} (n={len(values)})")
        rank_values = [item["rank_value"] for item in items if item["rank_value"] is not None]
        if rank_values:
            summary["_rank_value"] = mean(rank_values)
        summaries.append(summary)
        print()

    ranked = [item for item in summaries if "_rank_value" in item]
    ranked.sort(key=lambda item: item["_rank_value"], reverse=not args.rank_lower_is_better)
    if ranked:
        print(f"rank_metric: {args.rank_metric}")
        for rank, item in enumerate(ranked, start=1):
            print(f"  {rank}. {item['label']}: {fmt.format(item['_rank_value'])}")


def load_rows(
    paths: list[Path],
    group_keys: list[str],
    metric_keys: list[str],
    rank_metric: str,
) -> list[dict[str, Any]]:
    rows = []
    requested = set(metric_keys) | {rank_metric}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics: dict[str, float] = {}
        for metric in requested:
            try:
                value = get_nested(payload, metric)
            except KeyError:
                continue
            if value is None:
                continue
            metrics[metric] = float(value)
        group = tuple(get_nested(payload, key) for key in group_keys)
        rows.append(
            {
                "path": path,
                "group": group,
                "metrics": metrics,
                "rank_value": metrics.get(rank_metric),
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


if __name__ == "__main__":
    main()
