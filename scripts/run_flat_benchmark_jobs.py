from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


DEFAULT_ENVS = {
    "ant": "ant_full_adaptive",
    "halfcheetah": "halfcheetah_full_adaptive",
    "hopper": "hopper_full_adaptive",
    "inverted_double_pendulum": "inverteddoublependulum_full_adaptive",
    "inverteddoublependulum": "inverteddoublependulum_full_adaptive",
    "pusher": "pusher_full_adaptive",
    "swimmer": "swimmer_full_adaptive",
    "walker2d": "walker2d_full_adaptive",
}


@dataclass
class PendingJob:
    config_name: str
    model: str
    seed: str


@dataclass
class RunningJob:
    pending: PendingJob
    process: subprocess.Popen[str]
    log_file: object
    log_path: Path
    command: list[str]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run env x seed x model jobs with a flat global scheduler. "
            "Each model streams through environments independently, so fast "
            "baselines do not wait for slow methods at an environment barrier."
        )
    )
    parser.add_argument(
        "--envs",
        default=(
            "swimmer,hopper,walker2d,halfcheetah,"
            "inverted_double_pendulum,pusher,ant"
        ),
        help=(
            "Comma-separated env aliases/configs. Known aliases: "
            + ",".join(sorted(DEFAULT_ENVS))
            + ". You may also pass a Hydra config name directly."
        ),
    )
    parser.add_argument("--seeds", default="0")
    parser.add_argument(
        "--models",
        default="duc_wm,cadm_supervised,pets_context,cadm_context,lean_gr",
    )
    parser.add_argument("--max-parallel", type=int, default=5)
    parser.add_argument("--output-dir", default="outputs/workshop_7env_5methods_seed0")
    parser.add_argument("--log-root", default="outputs/balanced_logs/workshop_7env_5methods_seed0")
    parser.add_argument(
        "--allow-multiple-per-model",
        action="store_true",
        help=(
            "Allow more than one environment for the same model to run at once. "
            "Default keeps one active stream per model for balanced GPU use."
        ),
    )
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Extra Hydra overrides forwarded to every run.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    configs = [resolve_config(item) for item in split_csv(args.envs)]
    seeds = split_csv(args.seeds)
    models = split_csv(args.models)
    if not configs:
        raise SystemExit("no env configs requested")
    if not seeds:
        raise SystemExit("no seeds requested")
    if not models:
        raise SystemExit("no models requested")

    max_parallel = len(models) if args.max_parallel <= 0 else max(1, int(args.max_parallel))
    pending_by_model = {
        model: [PendingJob(config_name, model, seed) for seed in seeds for config_name in configs]
        for model in models
    }
    running: list[RunningJob] = []
    failures: list[tuple[PendingJob, int, Path]] = []
    started_at = time.time()

    env = os.environ.copy()
    src_path = str(repo_root / "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else src_path

    try:
        while has_pending(pending_by_model) or running:
            launch_ready_jobs(
                pending_by_model=pending_by_model,
                running=running,
                repo_root=repo_root,
                env=env,
                max_parallel=max_parallel,
                log_root=Path(args.log_root),
                output_dir=str(args.output_dir),
                overrides=list(args.overrides),
                allow_multiple_per_model=bool(args.allow_multiple_per_model),
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
                label = job_label(job.pending)
                status = "ok" if code == 0 else f"failed:{code}"
                print(f"[done] {label}: {status} log={job.log_path} elapsed_s={elapsed:.1f}", flush=True)
                if code != 0:
                    failures.append((job.pending, code, job.log_path))
            running = still_running
    except KeyboardInterrupt:
        print("[interrupt] terminating running jobs...", flush=True)
        for job in running:
            job.process.terminate()
            job.log_file.close()
        raise

    if failures:
        for pending, code, log_path in failures:
            print(f"[failure] {job_label(pending)} exited with {code}; see {log_path}", flush=True)
        raise SystemExit(1)
    print(f"[all_done] elapsed_s={time.time() - started_at:.1f}", flush=True)


def launch_ready_jobs(
    pending_by_model: dict[str, list[PendingJob]],
    running: list[RunningJob],
    repo_root: Path,
    env: dict[str, str],
    max_parallel: int,
    log_root: Path,
    output_dir: str,
    overrides: list[str],
    allow_multiple_per_model: bool,
) -> None:
    while len(running) < max_parallel:
        active_models = {job.pending.model for job in running}
        pending = next_pending_job(pending_by_model, active_models, allow_multiple_per_model)
        if pending is None:
            return
        log_path = log_root / pending.config_name / f"seed_{pending.seed}" / f"{pending.model}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("w", encoding="utf-8")
        hydra_dir = (
            f"outputs/hydra_flat/{pending.config_name}/"
            f"seed_{pending.seed}/{pending.model}"
        )
        command = [
            sys.executable,
            "main.py",
            "--config-name",
            pending.config_name,
            f"model={pending.model}",
            f"seed={pending.seed}",
            f"output_dir={output_dir}",
            f"hydra.run.dir={hydra_dir}",
            *overrides,
        ]
        print(f"[launch] {job_label(pending)}: {' '.join(command)}", flush=True)
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
                pending=pending,
                process=process,
                log_file=log_file,
                log_path=log_path,
                command=command,
            )
        )


def next_pending_job(
    pending_by_model: dict[str, list[PendingJob]],
    active_models: set[str],
    allow_multiple_per_model: bool,
) -> PendingJob | None:
    for model, jobs in pending_by_model.items():
        if not jobs:
            continue
        if not allow_multiple_per_model and model in active_models:
            continue
        return jobs.pop(0)
    return None


def has_pending(pending_by_model: dict[str, list[PendingJob]]) -> bool:
    return any(jobs for jobs in pending_by_model.values())


def split_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def resolve_config(raw: str) -> str:
    key = raw.strip().lower()
    return DEFAULT_ENVS.get(key, raw.strip())


def job_label(job: PendingJob) -> str:
    return f"{job.config_name}/seed_{job.seed}/{job.model}"


if __name__ == "__main__":
    main()
