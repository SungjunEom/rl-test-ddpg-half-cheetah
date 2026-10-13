# PPO and DDPG on HalfCheetah-v5

Train and compare the two algorithms with equal environment-step budgets. The
corrected implementation distinguishes termination from time-limit truncation,
and PPO's GAE stops at resets while bootstrapping the final pre-reset observation
at timeouts. Networks and optimization hyperparameters retain the original
configuration, including full-rollout PPO updates.

## Environment and tests

The `pytorch-docker` container mounts this repository at `/workspace`. Install
`requirements.txt` in a CUDA-compatible PyTorch environment, then run:

```sh
docker exec -w /workspace pytorch-docker python -m pytest -q
```

## Experiments

```sh
docker exec -w /workspace pytorch-docker python -u run_experiments.py
```

Defaults: 20 seeds per algorithm, exactly 1,000,000 training transitions per run,
four concurrent workers, and ten deterministic evaluation episodes every 50,000
steps (plus initial and final evaluation). Both algorithms use training seeds
1–20 and evaluation reset seeds 100000–100009. Evaluation uses a separate
environment and does not consume training random-number streams. Its steps do
not count toward the training budget. Seeds aid reproducibility; bitwise
reproducibility across hardware/library versions is not guaranteed.

The runner creates a fresh `runs/corrected_<timestamp>/` directory, saves stdout
and stderr per run, reports failed subprocesses, and generates `comparison_plot.png`
and `summary.json` only after all runs finish successfully. Each run stores its
configuration, TensorBoard events, evaluation returns, final actor weights, and
a completion marker. Existing runs are never overwritten. Rendering is optional
(`main.py --record`), so benchmark jobs do not require a display.

For a short end-to-end check:

```sh
python run_experiments.py --runs 1 --workers 2 --total_steps 2101 --eval_interval 1000 --eval_episodes 2
```

For a single run:

```sh
python main.py --algo ppo --seed 1 --total_steps 1000000 --output_dir runs/example
```

`--max_episodes` remains supported as a legacy way to specify a budget of
`max_episodes * 1000`; `--total_steps` takes precedence. Re-evaluate an actor with
`main.py --algo ppo --eval --model_path <checkpoint>`, optionally adding `--record`
for a GIF (headless rendering may require `MUJOCO_GL=egl`). Actor checkpoints are
for inference, not resuming optimizer/replay state.

## Comparison

```sh
python compare_runs.py --root runs/corrected_<timestamp> --num_runs 20
```

The comparison uses deterministic evaluation returns at matching training steps,
not exploratory training rewards. Final scores are each run's mean over the
final evaluation episodes; confidence intervals use the actual number of runs,
sample standard deviation, and Student's t critical value. With one run, no
confidence interval is estimated. Missing/incomplete runs or mismatched evaluation
schedules cause an error instead of a misleading partial comparison.
