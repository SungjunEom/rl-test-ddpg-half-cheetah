"""Compare deterministic evaluations at identical training-step checkpoints."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t


def mean_ci(data):
    data = np.asarray(data, dtype=float)
    n = data.shape[0]
    mean = data.mean(axis=0)
    # A single run cannot estimate uncertainty across independent runs.
    ci = np.full_like(mean, np.nan) if n < 2 else t.ppf(0.975, n - 1) * data.std(axis=0, ddof=1) / np.sqrt(n)
    return mean, ci


def get_all_runs_data(algo, num_runs=20, root="runs/corrected"):
    runs = []
    configs = []
    for run_id in range(1, num_runs + 1):
        path = Path(root) / f"{algo}_halfcheetah_run_{run_id}"
        if not (path / "completed.json").exists():
            raise ValueError(f"Missing or incomplete run: {path}")
        config = json.loads((path / "config.json").read_text())
        rows = [json.loads(line) for line in (path / "evaluations.jsonl").read_text().splitlines()]
        by_step = {row["step"]: row["mean"] for row in rows}
        if len(by_step) != len(rows) or config["total_steps"] not in by_step:
            raise ValueError(f"Duplicate or missing final evaluation: {path}")
        configs.append(config)
        runs.append(by_step)
    reference = set(runs[0])
    keys = ("total_steps", "eval_interval", "eval_episodes", "eval_seed")
    if any(set(run) != reference for run in runs) or any(
        any(config[key] != configs[0][key] for key in keys) for config in configs
    ):
        raise ValueError(f"Incompatible evaluation schedules for {algo}")
    steps = np.array(sorted(reference))
    data = np.array([[run[int(step)] for step in steps] for run in runs])
    mean, ci = mean_ci(data)
    return steps, mean, ci, data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="runs/corrected")
    parser.add_argument("--num_runs", type=int, default=20)
    args = parser.parse_args()
    if args.num_runs < 1:
        parser.error("--num_runs must be positive")
    results = {algo: get_all_runs_data(algo, args.num_runs, args.root) for algo in ("ddpg", "ppo")}
    if not np.array_equal(results["ddpg"][0], results["ppo"][0]):
        raise ValueError("Algorithms have different evaluation checkpoints")
    for run_id in range(1, args.num_runs + 1):
        configs = [json.loads((Path(args.root) / f"{algo}_halfcheetah_run_{run_id}" / "config.json").read_text()) for algo in results]
        if any(configs[0][key] != configs[1][key] for key in ("seed", "eval_seed", "eval_episodes", "total_steps")):
            raise ValueError(f"Algorithms have incompatible evaluation settings for run {run_id}")
    fig, (curves, bars) = plt.subplots(1, 2, figsize=(14, 5))
    summary = {}
    for index, (algo, (steps, mean, ci, data)) in enumerate(results.items()):
        curves.plot(steps, mean, label=f"{algo.upper()} (n={len(data)})")
        curves.fill_between(steps, mean-ci, mean+ci, alpha=0.2)
        final_mean, final_ci = mean_ci(data[:, -1])
        bars.bar(index, float(final_mean), yerr=float(final_ci) if len(data) > 1 else None, capsize=5)
        summary[algo] = dict(runs=len(data), training_steps=int(steps[-1]),
                             mean_return=float(final_mean), ci95_half_width=float(final_ci) if len(data) > 1 else None,
                             per_run_returns=data[:, -1].tolist())
    curves.set(xlabel="Training environment steps", ylabel="Deterministic evaluation return", title="HalfCheetah-v5: mean and 95% t interval")
    curves.legend()
    bars.set(xticks=[0, 1], xticklabels=["DDPG", "PPO"], ylabel="Deterministic evaluation return", title="Final checkpoint (10 evaluation episodes by default)")
    fig.tight_layout()
    fig.savefig(Path(args.root) / "comparison_plot.png")
    plt.close(fig)
    (Path(args.root) / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
