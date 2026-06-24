from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_ENVS = {
    "ant": "ant_full_adaptive",
    "halfcheetah": "halfcheetah_full_adaptive",
    "inverted_double_pendulum": "inverteddoublependulum_full_adaptive",
    "inverteddoublependulum": "inverteddoublependulum_full_adaptive",
    "swimmer": "swimmer_full_adaptive",
    "hopper": "hopper_full_adaptive",
    "pusher": "pusher_full_adaptive",
    "walker2d": "walker2d_full_adaptive",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the main DUC-WM workshop MuJoCo suite across env configs, "
            "seeds, and fair baseline methods."
        )
    )
    parser.add_argument(
        "--envs",
        default="swimmer,hopper,walker2d,halfcheetah,inverted_double_pendulum",
        help=(
            "Comma-separated env aliases/configs. Known aliases: "
            + ",".join(sorted(DEFAULT_ENVS))
            + ". You may also pass a Hydra config name directly."
        ),
    )
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--models", default="duc_wm,cadm_supervised,pets_context")
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument("--output-dir", default="outputs/duc_wm_workshop_main")
    parser.add_argument("--log-root", default="outputs/balanced_logs/workshop_main")
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Extra Hydra overrides forwarded to every run.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    configs = [resolve_config(item) for item in split_csv(args.envs)]
    seeds = split_csv(args.seeds)
    if not configs:
        raise SystemExit("no env configs requested")
    if not seeds:
        raise SystemExit("no seeds requested")

    for config_name in configs:
        for seed in seeds:
            log_dir = Path(args.log_root) / config_name / f"seed_{seed}"
            command = [
                sys.executable,
                "scripts/run_balanced_gpu_jobs.py",
                "--config-name",
                config_name,
                "--models",
                args.models,
                "--max-parallel",
                str(args.max_parallel),
                "--log-dir",
                str(log_dir),
                f"seed={seed}",
                f"output_dir={args.output_dir}",
                *args.overrides,
            ]
            print(f"[suite] {' '.join(command)}", flush=True)
            subprocess.run(command, cwd=repo_root, check=True)


def split_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def resolve_config(raw: str) -> str:
    key = raw.strip().lower()
    return DEFAULT_ENVS.get(key, raw.strip())


if __name__ == "__main__":
    main()
