"""Render matched rollout frames from final PPO/DDPG actor checkpoints."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MPLBACKEND", "Agg")

import gymnasium as gym
import matplotlib.pyplot as plt
import torch
from PIL import Image

from main import deterministic_action
from models.agent import DDPGAgent
from models.ppo_agent import PPOAgent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/full_cuda_20261015_182800"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/final_snapshots"))
    parser.add_argument("--run-id", type=int, default=1)
    parser.add_argument("--eval-seed", type=int, default=100000)
    parser.add_argument("--frame-step", type=int, default=50,
                        help="Common rollout timestep to capture (default: 50, or 2.5 s)")
    args = parser.parse_args()
    if args.frame_step < 1:
        parser.error("--frame-step must be positive")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    records, frames = [], []
    for algo, agent_class in (("ppo", PPOAgent), ("ddpg", DDPGAgent)):
        run = args.root / f"{algo}_halfcheetah_run_{args.run_id}"
        config = json.loads((run / "config.json").read_text())
        completed = json.loads((run / "completed.json").read_text())
        assert completed["total_steps"] == config["total_steps"]
        checkpoint = run / "actor.pth"
        env = gym.make("HalfCheetah-v5", render_mode="rgb_array", width=1200, height=800)
        try:
            agent = agent_class(env.observation_space.shape[0], env.action_space.shape[0],
                                float(env.action_space.high[0]), device="cpu")
            agent.actor.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            agent.actor.eval()
            state, _ = env.reset(seed=args.eval_seed)
            score, steps = 0.0, 0
            frame = None
            while True:
                state, reward, terminated, truncated, _ = env.step(deterministic_action(agent, algo, state))
                score += float(reward)
                steps += 1
                if steps == args.frame_step:
                    frame = env.render()
                    frame_time = float(env.unwrapped.data.time)
                    frame_root_x = float(env.unwrapped.data.qpos[0])
                if terminated or truncated:
                    break
            if frame is None:
                raise ValueError(f"Episode ended before requested frame {args.frame_step}")
            Image.fromarray(frame).save(args.output / f"{algo}_final.png")
            rows = [json.loads(line) for line in (run / "evaluations.jsonl").read_text().splitlines()]
            final = rows[-1]
            reference = final["returns"][final["seeds"].index(args.eval_seed)]
            record = dict(algorithm=algo.upper(), checkpoint=str(checkpoint.resolve()),
                          checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                          training_seed=config["seed"], training_steps=completed["total_steps"],
                          evaluation_seed=args.eval_seed, frame_step=args.frame_step,
                          simulation_time_seconds=frame_time, episode_steps=steps,
                          episode_return=score, saved_episode_return=reference,
                          return_difference=score-reference, terminated=bool(terminated), truncated=bool(truncated),
                          root_x=frame_root_x)
            records.append(record)
            frames.append(frame)
            print(json.dumps(record), flush=True)
        finally:
            env.close()
    assert records[0]["training_seed"] == records[1]["training_seed"]
    assert records[0]["training_steps"] == records[1]["training_steps"]
    assert records[0]["frame_step"] == records[1]["frame_step"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), facecolor="white")
    for ax, frame, record in zip(axes, frames, records):
        ax.imshow(frame)
        ax.set_title(record["algorithm"], fontsize=16, weight="bold")
        ax.axis("off")
    fig.suptitle("HalfCheetah-v5 | Final trained policies", fontsize=17)
    note = (f"{records[0]['training_steps']:,} training steps · training seed {records[0]['training_seed']} · "
            f"evaluation seed {args.eval_seed}\n"
            f"Deterministic actions · rollout step {records[0]['frame_step']} "
            f"({records[0]['simulation_time_seconds']:.1f} s) · identical tracking camera")
    fig.text(0.5, 0.04, note, ha="center", fontsize=10, linespacing=1.6)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.87, bottom=0.17, wspace=0.03)
    fig.savefig(args.output / "ppo_ddpg_final_comparison.png", dpi=250)
    plt.close(fig)
    metadata = dict(environment="HalfCheetah-v5", inference_device="cpu",
                    provenance_note="Fresh rollouts from saved final actors, not original GPU evaluation frames.",
                    torch_version=torch.__version__,
                    gymnasium_version=gym.__version__, frame_selection="Fixed common intermediate timestep selected before rendering; no pose selection",
                    camera="Identical Gymnasium default tracking camera; world position is not aligned", records=records)
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    caption = (f"Final-policy snapshots of PPO (left) and DDPG (right) in HalfCheetah-v5 after "
               f"{records[0]['training_steps']:,} training transitions. Both use training seed "
               f"{records[0]['training_seed']}, deterministic actions, evaluation reset seed {args.eval_seed}, "
               f"and the same camera settings. Frames are taken after transition {records[0]['frame_step']} "
               f"({records[0]['simulation_time_seconds']:.1f} s) in each evaluation rollout. Final refers to "
               "the training checkpoint, not the last episode frame. The common timestep was selected "
               "before rendering. These single-run illustrations show instantaneous posture, not gait, "
               "speed, or performance across seeds.\n")
    (args.output / "caption.txt").write_text(caption)


if __name__ == "__main__":
    main()
