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
    parser.add_argument("--output_prefix", default="comparison_plot", help="Figure filename stem within --root")
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
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "axes.linewidth": 0.8,
        "grid.alpha": 0.2,
        "grid.linestyle": "--",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "mathtext.fontset": "dejavuserif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, (curves, bars) = plt.subplots(
        1, 2, figsize=(7.2, 3.25), gridspec_kw={"width_ratios": [1.65, 1]},
        sharey=True,
    )
    colors = {"ddpg": "#0072B2", "ppo": "#D55E00"}
    for ax in (curves, bars):
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y")
        ax.tick_params(direction="out", width=0.8, length=3)
        ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    summary = {}
    for index, (algo, (steps, mean, ci, data)) in enumerate(results.items()):
        color = colors[algo]
        curves.plot(steps / 1e6, mean, color=color, linewidth=1.6,
                    linestyle="-" if algo == "ddpg" else "--", label=algo.upper())
        curves.fill_between(steps / 1e6, mean-ci, mean+ci, color=color, alpha=0.16, linewidth=0)
        final_mean, final_ci = mean_ci(data[:, -1])
        bars.bar(index, float(final_mean), width=0.58, color=color, alpha=0.85,
                 yerr=float(final_ci) if len(data) > 1 else None,
                 error_kw={"elinewidth": 0.9, "capthick": 0.9, "capsize": 3})
        bars.annotate(f"{float(final_mean):,.0f}",
                      (index, float(final_mean) + (float(final_ci) if len(data) > 1 else 0)),
                      xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8)
        summary[algo] = dict(runs=len(data), training_steps=int(steps[-1]),
                             mean_return=float(final_mean), ci95_half_width=float(final_ci) if len(data) > 1 else None,
                             per_run_returns=data[:, -1].tolist())
    curves.set(xlabel="Training environment steps (millions)", ylabel="Deterministic evaluation return",
               title="(a) Learning curves", xlim=(0, steps[-1] / 1e6))
    curves.legend(frameon=False, loc="upper left")
    bars.set(xticks=[0, 1], xticklabels=["DDPG", "PPO"], title="(b) Final performance", xlim=(-0.6, 1.6))
    curves.margins(y=0.14)
    fig.suptitle("HalfCheetah-v5", y=0.98, fontsize=10)
    episodes = configs[0]["eval_episodes"]
    note = (f"{args.num_runs} seeds per algorithm; {episodes} evaluation episodes per checkpoint. "
            + ("Shading / error bars: 95% t confidence intervals." if args.num_runs > 1 else "Confidence intervals unavailable for one seed."))
    fig.text(0.5, 0.025, note, ha="center", fontsize=7)
    fig.tight_layout(rect=(0, 0.08, 1, 0.96), w_pad=1.8)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(Path(args.root) / f"{args.output_prefix}.{extension}", bbox_inches="tight")
    plt.close(fig)
    (Path(args.root) / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
