import argparse
import json
import random
from pathlib import Path

import gymnasium as gym
import imageio
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from models.agent import DDPGAgent
from models.ppo_agent import PPOAgent
from models.replay_buffer import ReplayBuffer
from models.rollout_buffer import RolloutBuffer


def deterministic_action(agent, algo, state):
    action = agent.select_action(state, noise_scale=0.0) if algo == "ddpg" else agent.evaluate(state)
    return np.clip(action, -agent.max_action, agent.max_action)


def evaluate(agent, algo, env, seeds):
    """Separate environment and fixed reset seeds; do not consume training RNGs."""
    scores = []
    for seed in seeds:
        state, _ = env.reset(seed=seed)
        score = 0.0
        while True:
            state, reward, terminated, truncated, _ = env.step(deterministic_action(agent, algo, state))
            score += reward
            if terminated or truncated:
                break
        scores.append(float(score))
    return scores


def evaluate_and_record(agent, algo, env_id="HalfCheetah-v5", filepath="eval_rollout.gif", seed=100000):
    env = gym.make(env_id, render_mode="rgb_array")
    try:
        state, _ = env.reset(seed=seed)
        frames = [env.render()]
        score = 0.0
        while True:
            state, reward, terminated, truncated, _ = env.step(deterministic_action(agent, algo, state))
            score += reward
            frames.append(env.render())
            if terminated or truncated:
                break
        imageio.mimsave(filepath, frames, fps=30)
        print(f"Recorded evaluation reward: {score:.2f} -> {filepath}", flush=True)
    finally:
        env.close()


def parse_args():
    parser = argparse.ArgumentParser(description="PPO/DDPG on HalfCheetah-v5")
    parser.add_argument("--algo", default="ddpg", choices=["ddpg", "ppo"])
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_episodes", type=int, default=1000, help="Legacy budget: 1000 steps per episode")
    parser.add_argument("--total_steps", type=int, default=None, help="Exact training transition budget for either algorithm")
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--noise_scale", type=float, default=0.1)
    parser.add_argument("--capacity", type=int, default=1000000)
    parser.add_argument("--ppo_steps", type=int, default=2048)
    parser.add_argument("--ppo_epochs", type=int, default=10)
    parser.add_argument("--clip_ratio", type=float, default=0.2)
    parser.add_argument("--run_id", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--eval_interval", type=int, default=50000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--eval_seed", type=int, default=100000)
    parser.add_argument("--output_dir", default="runs/corrected")
    parser.add_argument("--record", action="store_true", help="Also render a final GIF")
    args = parser.parse_args()
    args.total_steps = args.total_steps if args.total_steps is not None else args.max_episodes * 1000
    args.seed = args.seed if args.seed is not None else args.run_id
    for field in ("total_steps", "batch_size", "capacity", "ppo_steps", "ppo_epochs", "threads", "eval_interval", "eval_episodes"):
        if getattr(args, field) <= 0:
            parser.error(f"--{field} must be positive")
    if args.algo == "ddpg" and args.capacity < args.batch_size:
        parser.error("--capacity must be at least --batch_size")
    return args


def train(agent, args, env, writer, on_evaluate):
    """Train to an exact transition count; evaluations never reset the training env."""
    if args.algo == "ddpg":
        buffer = ReplayBuffer(args.capacity, env.observation_space.shape[0], env.action_space.shape[0], agent.device)
    else:
        buffer = RolloutBuffer(args.ppo_steps, env.observation_space.shape[0], env.action_space.shape[0], agent.device)
    state, _ = env.reset(seed=args.seed)
    episode_reward = 0.0
    episode = 0
    for step in range(1, args.total_steps + 1):
        if args.algo == "ddpg":
            action = agent.select_action(state, noise_scale=args.noise_scale)
            next_state, reward, terminated, truncated, _ = env.step(action)
            buffer.push(state, action, reward, next_state, terminated)
            if buffer.size >= args.batch_size:
                critic_loss, actor_loss = agent.train(buffer, args.batch_size)
                if step % 100 == 0:
                    writer.add_scalar("Loss/Critic", critic_loss, step)
                    writer.add_scalar("Loss/Actor", actor_loss, step)
        else:
            action, logprob, value = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(np.clip(action, -agent.max_action, agent.max_action))
            buffer.push(state, action, logprob, reward, value, next_state, terminated, truncated)
            # Keep rollout boundaries independent of the evaluation schedule.
            if buffer.ptr == buffer.capacity or step == args.total_steps:
                critic_loss, actor_loss = agent.train(buffer)
                writer.add_scalar("Loss/Critic", critic_loss, step)
                writer.add_scalar("Loss/Actor", actor_loss, step)
        state = next_state
        episode_reward += reward
        if terminated or truncated:
            episode += 1
            writer.add_scalar("Reward/Episode", episode_reward, episode)
            writer.add_scalar("Reward/Training", episode_reward, step)
            print(f"{args.algo} run {args.run_id} step {step} episode {episode} reward {episode_reward:.2f}", flush=True)
            state, _ = env.reset()
            episode_reward = 0.0
        if step % args.eval_interval == 0 or step == args.total_steps:
            on_evaluate(step)


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env_id = "HalfCheetah-v5"
    env = gym.make(env_id)
    eval_env = gym.make(env_id)
    writer = None
    try:
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
        if args.algo == "ddpg":
            agent = DDPGAgent(state_dim, action_dim, max_action, lr=args.lr if args.lr is not None else 1e-3,
                              gamma=args.gamma, tau=args.tau, device=args.device)
        else:
            agent = PPOAgent(state_dim, action_dim, max_action, lr=args.lr if args.lr is not None else 3e-4,
                             gamma=args.gamma, clip_ratio=args.clip_ratio, ppo_epochs=args.ppo_epochs, device=args.device)
        run_dir = Path(args.output_dir) / f"{args.algo}_halfcheetah_run_{args.run_id}"
        model_path = Path(args.model_path) if args.model_path else run_dir / "actor.pth"
        seeds = list(range(args.eval_seed, args.eval_seed + args.eval_episodes))
        print(f"{args.algo.upper()} | device={agent.device} seed={args.seed} steps={args.total_steps}", flush=True)
        if args.eval:
            agent.actor.load_state_dict(torch.load(model_path, map_location=agent.device, weights_only=True))
            scores = evaluate(agent, args.algo, eval_env, seeds)
            print(f"Evaluation: mean={np.mean(scores):.2f}, returns={scores}", flush=True)
        else:
            # Never merge a new training run into existing event files.
            run_dir.mkdir(parents=True, exist_ok=False)
            config = dict(vars(args), actual_device=str(agent.device), torch_version=torch.__version__,
                          gymnasium_version=gym.__version__, numpy_version=np.__version__)
            (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
            writer = SummaryWriter(str(run_dir))
            def record_evaluation(step):
                scores = evaluate(agent, args.algo, eval_env, seeds)
                mean = float(np.mean(scores))
                writer.add_scalar("Eval/MeanReturn", mean, step)
                writer.flush()
                with (run_dir / "evaluations.jsonl").open("a") as stream:
                    stream.write(json.dumps(dict(step=step, seeds=seeds, returns=scores, mean=mean)) + "\n")
                print(f"EVAL {args.algo} run {args.run_id} step {step}: {mean:.2f}", flush=True)
            record_evaluation(0)
            train(agent, args, env, writer, record_evaluation)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(agent.actor.state_dict(), model_path)
            (run_dir / "completed.json").write_text(json.dumps(dict(total_steps=args.total_steps, seed=args.seed)) + "\n")
        if args.record:
            evaluate_and_record(agent, args.algo, env_id, str(model_path.with_suffix(".gif")), args.eval_seed)
    finally:
        if writer is not None:
            writer.close()
        env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
