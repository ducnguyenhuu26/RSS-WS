from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any


DEFAULT_METRIC = "score.duc_r2_at_1"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate MuJoCo JSON outputs across seeds."
    )
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    parser.add_argument("--precision", type=int, default=4)
    parser.add_argument("--show-std", action="store_true")
    args = parser.parse_args()

    rows = []
    for path in expand_input_paths(args.files):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "path": path,
                "problem": payload.get("problem") or payload.get("env_id"),
                "model": payload.get("model"),
                "seed": payload.get("seed"),
                "value": float(get_nested(payload, args.metric)),
            }
        )

    if not rows:
        raise SystemExit("no readable result rows")

    values = [row["value"] for row in rows]
    avg = mean(values)
    spread = stdev(values) if len(values) > 1 else 0.0
    fmt = f"{{:.{args.precision}f}}"
    problem = rows[0]["problem"]
    model = rows[0]["model"]

    print(f"metric: {args.metric}")
    print(f"problem: {problem}")
    print(f"model: {model}")
    print("seeds:")
    for row in sorted(rows, key=lambda item: item["seed"]):
        print(f"  seed={row['seed']}: {fmt.format(row['value'])} ({row['path']})")
    print(f"mean: {fmt.format(avg)}")
    print(f"std:  {fmt.format(spread)}")
    if args.show_std:
        print(f"COPY_TO_TABLE: {fmt.format(avg)} +/- {fmt.format(spread)}")
    else:
        print(f"COPY_TO_TABLE: {fmt.format(avg)}")


def get_nested(payload: dict[str, Any], dotted_key: str) -> Any:
    current: Any = payload
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"metric '{dotted_key}' not found; missing '{part}'")
        current = current[part]
    return current


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


if __name__ == "__main__":
    main()
