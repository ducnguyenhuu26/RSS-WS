from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any


DEFAULT_BASELINES = ("pets_context", "cadm_context", "cadm_supervised")
DEFAULT_HORIZONS = (1, 10, 25)


@dataclass(frozen=True)
class CandidateSummary:
    group: tuple[str, str]
    model: str
    n: int
    reward: float
    r2: dict[int, float]
    duc_r2: dict[int, float]
    ood: float | None
    risk_gap: float
    lower_bound: float
    dynamics_valid: bool
    ood_valid: bool
    valid: bool
    invalid_reasons: tuple[str, ...]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select planning candidates with baseline-relative dynamics guards, "
            "planner coverage diagnostics, and an oracle-style finite-candidate penalty."
        )
    )
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--baselines", default=",".join(DEFAULT_BASELINES))
    parser.add_argument("--horizons", default=",".join(str(item) for item in DEFAULT_HORIZONS))
    parser.add_argument("--r2-tolerance", type=float, default=0.02)
    parser.add_argument("--duc-r2-tolerance", type=float, default=0.02)
    parser.add_argument("--ood-tolerance", type=float, default=0.25)
    parser.add_argument("--risk-weight", type=float, default=1.0)
    parser.add_argument("--ood-weight", type=float, default=1.0)
    parser.add_argument("--score-range", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=0.10)
    parser.add_argument("--allow-missing-baselines", action="store_true")
    parser.add_argument("--csv-out", type=Path)
    args = parser.parse_args()

    baselines = tuple(item.strip() for item in args.baselines.split(",") if item.strip())
    horizons = tuple(int(item.strip()) for item in args.horizons.split(",") if item.strip())
    payloads = load_payloads(expand_input_paths(args.files))
    if not payloads:
        raise SystemExit("no JSON payloads found")
    summaries = summarize_candidates(
        payloads=payloads,
        baselines=baselines,
        horizons=horizons,
        r2_tolerance=float(args.r2_tolerance),
        duc_r2_tolerance=float(args.duc_r2_tolerance),
        ood_tolerance=float(args.ood_tolerance),
        risk_weight=float(args.risk_weight),
        ood_weight=float(args.ood_weight),
        score_range=float(args.score_range),
        delta=float(args.delta),
        allow_missing_baselines=bool(args.allow_missing_baselines),
    )
    print_summaries(summaries, horizons)
    if args.csv_out is not None:
        write_csv(args.csv_out, summaries, horizons)


def expand_input_paths(paths: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        matches = [Path(item) for item in glob.glob(str(path))]
        expanded.extend(matches if matches else [path])
    return sorted(set(expanded))


def load_payloads(paths: list[Path]) -> list[dict[str, Any]]:
    payloads = []
    for path in paths:
        if path.is_file():
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
    return payloads


def summarize_candidates(
    payloads: list[dict[str, Any]],
    baselines: tuple[str, ...],
    horizons: tuple[int, ...],
    r2_tolerance: float,
    duc_r2_tolerance: float,
    ood_tolerance: float,
    risk_weight: float,
    ood_weight: float,
    score_range: float,
    delta: float,
    allow_missing_baselines: bool,
) -> list[CandidateSummary]:
    grouped: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for payload in payloads:
        group = (str(payload.get("problem", "")), str(payload.get("test_variant", "")))
        grouped[group][str(payload.get("model", ""))].append(payload)

    summaries: list[CandidateSummary] = []
    for group, by_model in sorted(grouped.items()):
        baseline_rows = {
            model: rows for model, rows in by_model.items() if model in baselines
        }
        baseline_best_r2 = {
            horizon: max(
                (
                    metric_mean(rows, f"r2_at_{horizon}")
                    for rows in baseline_rows.values()
                    if metric_mean(rows, f"r2_at_{horizon}") is not None
                ),
                default=None,
            )
            for horizon in horizons
        }
        baseline_best_duc_r2 = {
            horizon: max(
                (
                    metric_mean(rows, f"duc_r2_at_{horizon}")
                    for rows in baseline_rows.values()
                    if metric_mean(rows, f"duc_r2_at_{horizon}") is not None
                ),
                default=None,
            )
            for horizon in horizons
        }
        baseline_ood = min(
            (
                value
                for rows in baseline_rows.values()
                if (value := metric_mean(rows, "planner_ood_mean")) is not None
            ),
            default=None,
        )
        k = max(1, len(by_model))
        n = max(1, min(len(rows) for rows in by_model.values()))
        rad = score_range * math.sqrt(math.log(2.0 * k / max(1e-12, delta)) / (2.0 * n))
        for model, rows in sorted(by_model.items()):
            r2 = {horizon: metric_mean(rows, f"r2_at_{horizon}") for horizon in horizons}
            duc_r2 = {horizon: metric_mean(rows, f"duc_r2_at_{horizon}") for horizon in horizons}
            reward = metric_mean(rows, "planner_return_mean") or 0.0
            ood = metric_mean(rows, "planner_ood_mean")
            risk_gap = metric_mean(rows, "certified_risk_optimistic_gap_mean") or 0.0
            reasons: list[str] = []
            if not baseline_rows and not allow_missing_baselines:
                reasons.append("no_baseline")
            for horizon in horizons:
                if baseline_best_r2[horizon] is not None and r2[horizon] is not None:
                    if r2[horizon] < baseline_best_r2[horizon] - r2_tolerance:
                        reasons.append(f"r2@{horizon}")
                if baseline_best_duc_r2[horizon] is not None and duc_r2[horizon] is not None:
                    if duc_r2[horizon] < baseline_best_duc_r2[horizon] - duc_r2_tolerance:
                        reasons.append(f"duc_r2@{horizon}")
            ood_valid = True
            if baseline_ood is not None and ood is not None:
                ood_valid = ood <= baseline_ood + ood_tolerance
                if not ood_valid:
                    reasons.append("planner_ood")
            ood_penalty = 0.0 if ood is None else ood_weight * ood
            lower_bound = reward - risk_weight * risk_gap - ood_penalty - rad
            dynamics_valid = not any(reason.startswith("r2@") or reason.startswith("duc_r2@") for reason in reasons)
            valid = not reasons
            summaries.append(
                CandidateSummary(
                    group=group,
                    model=model,
                    n=len(rows),
                    reward=reward,
                    r2={h: float(r2[h]) for h in horizons if r2[h] is not None},
                    duc_r2={h: float(duc_r2[h]) for h in horizons if duc_r2[h] is not None},
                    ood=ood,
                    risk_gap=risk_gap,
                    lower_bound=lower_bound,
                    dynamics_valid=dynamics_valid,
                    ood_valid=ood_valid,
                    valid=valid,
                    invalid_reasons=tuple(dict.fromkeys(reasons)),
                )
            )
    return summaries


def metric_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = []
    for row in rows:
        value = row.get("score", {}).get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    if not values:
        return None
    return float(mean(values))


def print_summaries(summaries: list[CandidateSummary], horizons: tuple[int, ...]) -> None:
    by_group: dict[tuple[str, str], list[CandidateSummary]] = defaultdict(list)
    for summary in summaries:
        by_group[summary.group].append(summary)
    for group, items in sorted(by_group.items()):
        print(f"{group[0]} | {group[1]}")
        ranked = sorted(items, key=lambda item: item.lower_bound, reverse=True)
        for rank, item in enumerate(ranked, start=1):
            r2_bits = " ".join(
                f"R2@{h}={item.r2.get(h, float('nan')):.4f}" for h in horizons
            )
            ood = "NA" if item.ood is None else f"{item.ood:.4f}"
            reasons = "ok" if item.valid else ",".join(item.invalid_reasons)
            print(
                f"  {rank}. {item.model}: valid={item.valid} lb={item.lower_bound:.4f} "
                f"reward={item.reward:.4f} ood={ood} risk_gap={item.risk_gap:.4f} "
                f"{r2_bits} reason={reasons}"
            )
        valid_items = [item for item in ranked if item.valid]
        selected = valid_items[0] if valid_items else None
        if selected is None:
            print("  selected: NONE (no candidate passed dynamics/coverage guards)")
        else:
            print(f"  selected: {selected.model}")
        print()


def write_csv(path: Path, summaries: list[CandidateSummary], horizons: tuple[int, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "problem",
        "test_variant",
        "model",
        "n",
        "valid",
        "lower_bound",
        "reward",
        "planner_ood_mean",
        "certified_risk_optimistic_gap_mean",
        "invalid_reasons",
    ]
    fields.extend(f"r2_at_{horizon}" for horizon in horizons)
    fields.extend(f"duc_r2_at_{horizon}" for horizon in horizons)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in summaries:
            row: dict[str, Any] = {
                "problem": item.group[0],
                "test_variant": item.group[1],
                "model": item.model,
                "n": item.n,
                "valid": item.valid,
                "lower_bound": item.lower_bound,
                "reward": item.reward,
                "planner_ood_mean": item.ood,
                "certified_risk_optimistic_gap_mean": item.risk_gap,
                "invalid_reasons": ",".join(item.invalid_reasons),
            }
            for horizon in horizons:
                row[f"r2_at_{horizon}"] = item.r2.get(horizon)
                row[f"duc_r2_at_{horizon}"] = item.duc_r2.get(horizon)
            writer.writerow(row)


if __name__ == "__main__":
    main()
