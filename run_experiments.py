"""Run isolated, seeded experiments and compare only after all runs succeed."""
import argparse
import concurrent.futures
import datetime
import json
import importlib.metadata
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time


def run_experiment(algo, run_id, args, root):
    cmd = [sys.executable, "-u", "main.py", "--algo", algo, "--run_id", str(run_id),
           "--seed", str(run_id), "--total_steps", str(args.total_steps),
           "--eval_interval", str(args.eval_interval), "--eval_episodes", str(args.eval_episodes),
           "--device", args.device, "--output_dir", str(root)]
    logfile = root / f"{algo}_run_{run_id}.log"
    print(f"Starting {algo} run {run_id}; log={logfile}", flush=True)
    with logfile.open("w") as stream:
        subprocess.run(cmd, check=True, stdout=stream, stderr=subprocess.STDOUT,
                       env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1"))
    print(f"Completed {algo} run {run_id}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--total_steps", type=int, default=1000000)
    parser.add_argument("--eval_interval", type=int, default=50000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()
    for field in ("runs", "workers", "total_steps", "eval_interval", "eval_episodes"):
        if getattr(args, field) <= 0:
            parser.error(f"--{field} must be positive")
    os.chdir(Path(__file__).resolve().parent)
    root = Path(args.output_dir or f"runs/corrected_{datetime.datetime.now():%Y%m%d_%H%M%S}").resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "experiment.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    # Keep the exact source and dependency versions alongside expensive runs.
    source_dir = root / "source"
    source_dir.mkdir()
    for filename in ("main.py", "run_experiments.py", "compare_runs.py", "requirements.txt"):
        shutil.copy2(filename, source_dir / filename)
    shutil.copytree("models", source_dir / "models", ignore=shutil.ignore_patterns("__pycache__"))
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "gymnasium", "mujoco", "numpy", "scipy", "tensorboard", "matplotlib", "imageio")}
    (root / "versions.json").write_text(json.dumps(versions, indent=2) + "\n")
    print(f"Experiment output: {root}", flush=True)
    started = time.monotonic()
    failures = []
    finished = 0
    def save_status(state):
        status = dict(state=state, finished_runs=finished, total_runs=2*args.runs,
                      elapsed_seconds=time.monotonic()-started, failures=failures,
                      completed=state == "completed")
        temporary = root / "status.tmp"
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(root / "status.json")
    save_status("running")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        tasks = {executor.submit(run_experiment, algo, run_id, args, root): (algo, run_id)
                 for run_id in range(1, args.runs + 1) for algo in ("ddpg", "ppo")}
        for future in concurrent.futures.as_completed(tasks):
            try:
                future.result()
            except Exception as exc:
                failures.append(dict(algo=tasks[future][0], run_id=tasks[future][1], error=str(exc)))
                print(f"FAILED {tasks[future]}: {exc}", flush=True)
            finished += 1
            save_status("running")
    if failures:
        save_status("failed")
        raise SystemExit(f"{len(failures)} runs failed; see logs in {root}")
    try:
        subprocess.run([sys.executable, "compare_runs.py", "--root", str(root), "--num_runs", str(args.runs)], check=True)
    except subprocess.CalledProcessError:
        save_status("comparison_failed")
        raise
    save_status("completed")
    print(f"All experiments and comparison completed in {time.monotonic()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
