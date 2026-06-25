from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunningJob:
    model: str
    process: subprocess.Popen[str]
    log_file: object
    log_path: Path
    command: list[str]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run SimFutures-LP benchmark methods as controlled parallel processes. "
            "This overlaps CPU-heavy MuJoCo collection/env stepping with "
            "GPU-heavy training/planning without oversubscribing one GPU too hard."
        )
    )
    parser.add_argument("--config-name", default="pilot_swimmer_gpu")
    parser.add_argument("--models", default="mlp,pets,cadm,duc_wm")
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Extra Hydra overrides, for example runtime.precision=fp16 planning.enabled=false",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    if not models:
        raise SystemExit("no models requested")
    max_parallel = len(models) if args.max_parallel <= 0 else min(args.max_parallel, len(models))
    log_dir = Path(args.log_dir) if args.log_dir else repo_root / "outputs" / "balanced_logs" / args.config_name
    log_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    src_path = str(repo_root / "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else src_path

    pending = list(models)
    running: list[RunningJob] = []
    failures: list[tuple[str, int, Path]] = []
    started_at = time.time()

    try:
        while pending or running:
            while pending and len(running) < max_parallel:
                model = pending.pop(0)
                log_path = log_dir / f"{model}.log"
                log_file = log_path.open("w", encoding="utf-8")
                hydra_dir = f"outputs/hydra_balanced/{args.config_name}/{model}"
                command = [
                    sys.executable,
                    "main.py",
                    "--config-name",
                    args.config_name,
                    f"model={model}",
                    f"hydra.run.dir={hydra_dir}",
                    *args.overrides,
                ]
                print(f"[launch] {model}: {' '.join(command)}")
                process = subprocess.Popen(
                    command,
                    cwd=repo_root,
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                running.append(
                    RunningJob(
                        model=model,
                        process=process,
                        log_file=log_file,
                        log_path=log_path,
                        command=command,
                    )
                )

            time.sleep(2.0)
            still_running: list[RunningJob] = []
            for job in running:
                code = job.process.poll()
                if code is None:
                    still_running.append(job)
                    continue
                job.log_file.close()
                elapsed = time.time() - started_at
                status = "ok" if code == 0 else f"failed:{code}"
                print(f"[done] {job.model}: {status} log={job.log_path} elapsed_s={elapsed:.1f}")
                if code != 0:
                    failures.append((job.model, code, job.log_path))
            running = still_running
    except KeyboardInterrupt:
        print("[interrupt] terminating running jobs...")
        for job in running:
            job.process.terminate()
            job.log_file.close()
        raise

    if failures:
        for model, code, log_path in failures:
            print(f"[failure] {model} exited with {code}; see {log_path}")
        raise SystemExit(1)
    print(f"[all_done] models={','.join(models)} elapsed_s={time.time() - started_at:.1f}")


if __name__ == "__main__":
    main()
