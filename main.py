import argparse
import gymnasium as gym
import numpy as np
import torch
import imageio
from torch.utils.tensorboard import SummaryWriter

from models.agent import DDPGAgent
from models.ppo_agent import PPOAgent
from models.replay_buffer import ReplayBuffer
from models.rollout_buffer import RolloutBuffer

def save_model(agent, filepath, algo):
    torch.save(agent.actor.state_dict(), filepath)
    print(f"\n[Info] Final {algo.upper()} model saved successfully to: {filepath}")

def evaluate_and_record(agent, algo, env_id="HalfCheetah-v5", filepath="eval_rollout.gif"):
    print(f"\n[Info] Evaluating agent deterministically and recording rollout...")
    env = gym.make(env_id, render_mode="rgb_array")
    
    raw_frames = []
    state, info = env.reset()
    raw_frames.append(env.render())
    
    episode_reward = 0
    done = False
    
    while not done:
        if algo == "ddpg":
            action = agent.select_action(state, noise_scale=0.0)
        else:
            action = agent.evaluate(state)
            action = np.clip(action, -agent.max_action, agent.max_action)
            
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        raw_frames.append(env.render())
        
    print(f"[Info] Evaluation Episode Reward: {episode_reward:.2f}")
    imageio.mimsave(filepath, raw_frames, fps=30)
    print(f"[Info] Evaluation video saved successfully to: {filepath}")
    env.close()

def main():
    parser = argparse.ArgumentParser(description="RL on HalfCheetah-v5")
    parser.add_argument("--algo", type=str, default="ddpg", choices=["ddpg", "ppo"], help="Algorithm to use (ddpg or ppo)")
    parser.add_argument("--eval", action="store_true", help="Only evaluate the saved model instead of training")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the saved actor model checkpoint")

    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--batch_size", type=int, default=256, help="Mini-batch size (DDPG)")
    parser.add_argument("--max_episodes", type=int, default=1000, help="Maximum number of training episodes")
    
    # DDPG specific
    parser.add_argument("--tau", type=float, default=0.005, help="Soft update target network coefficient (DDPG)")
    parser.add_argument("--noise_scale", type=float, default=0.1, help="Exploration noise (DDPG)")
    parser.add_argument("--capacity", type=int, default=1000000, help="Replay buffer capacity (DDPG)")
    
    # PPO specific
    parser.add_argument("--ppo_steps", type=int, default=2048, help="Rollout steps per update (PPO)")
    parser.add_argument("--ppo_epochs", type=int, default=10, help="PPO update epochs")
    parser.add_argument("--clip_ratio", type=float, default=0.2, help="PPO clip ratio")
    
    args = parser.parse_args()
    
    if args.model_path is None:
        args.model_path = f"{args.algo}_actor.pth"
    log_dir = f"runs/{args.algo}_halfcheetah"

    env_id = "HalfCheetah-v5"
    env = gym.make(env_id)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    if args.algo == "ddpg":
        lr = args.lr if args.lr is not None else 1e-3
        agent = DDPGAgent(state_dim, action_dim, max_action, lr=lr, gamma=args.gamma, tau=args.tau)
    else:
        lr = args.lr if args.lr is not None else 3e-4
        agent = PPOAgent(state_dim, action_dim, max_action, lr=lr, gamma=args.gamma, clip_ratio=args.clip_ratio, ppo_epochs=args.ppo_epochs)

    print(f"[Info] Algorithm: {args.algo.upper()} | Training on device: {agent.device}")

    if args.eval:
        try:
            agent.actor.load_state_dict(torch.load(args.model_path, map_location=agent.device))
            print(f"[Info] Successfully loaded model weights from: {args.model_path}")
            evaluate_and_record(agent, args.algo, env_id=env_id, filepath=f"{args.algo}_eval_rollout.gif")
        except FileNotFoundError:
            print(f"[Error] Could not find saved model weights at: {args.model_path}")
    else:
        writer = SummaryWriter(log_dir=log_dir)
        
        if args.algo == "ddpg":
            replay_buffer = ReplayBuffer(capacity=args.capacity, state_dim=state_dim, action_dim=action_dim, device=agent.device)
            global_step = 0
            for episode in range(args.max_episodes):
                state, info = env.reset()
                episode_reward = 0
                done = False
                
                while not done:
                    global_step += 1
                    action = agent.select_action(state, noise_scale=args.noise_scale)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    
                    replay_buffer.push(state, action, reward, next_state, done)
                    state = next_state
                    episode_reward += reward
                    
                    if replay_buffer.size > args.batch_size:
                        critic_loss, actor_loss = agent.train(replay_buffer, args.batch_size)
                        writer.add_scalar("Loss/Critic", critic_loss, global_step)
                        writer.add_scalar("Loss/Actor", actor_loss, global_step)
                        
                print(f"Episode: {episode + 1}, Reward: {episode_reward:.2f}")
                writer.add_scalar("Reward/Episode", episode_reward, episode + 1)
        else:
            rollout_buffer = RolloutBuffer(capacity=args.ppo_steps, state_dim=state_dim, action_dim=action_dim, device=agent.device)
            global_step = 0
            episode = 0
            
            # PPO training loop based on total timesteps to match max_episodes (approx 1000 steps per ep)
            total_timesteps = args.max_episodes * 1000 
            state, info = env.reset()
            episode_reward = 0
            
            while global_step < total_timesteps:
                for step in range(args.ppo_steps):
                    global_step += 1
                    action, logprob, value = agent.select_action(state)
                    
                    env_action = np.clip(action, -max_action, max_action)
                    next_state, reward, terminated, truncated, _ = env.step(env_action)
                    done = terminated or truncated
                    
                    rollout_buffer.push(state, action, logprob, reward, value, done)
                    
                    state = next_state
                    episode_reward += reward
                    
                    if done:
                        episode += 1
                        print(f"Episode: {episode}, Reward: {episode_reward:.2f}, Step: {global_step}")
                        writer.add_scalar("Reward/Episode", episode_reward, episode)
                        state, info = env.reset()
                        episode_reward = 0
                        
                critic_loss, actor_loss = agent.train(rollout_buffer, state, done)
                writer.add_scalar("Loss/Critic", critic_loss, global_step)
                writer.add_scalar("Loss/Actor", actor_loss, global_step)
                
                if episode >= args.max_episodes:
                    break

        save_model(agent, args.model_path, args.algo)
        evaluate_and_record(agent, args.algo, env_id=env_id, filepath=f"{args.algo}_eval_rollout.gif")
        writer.close()
    
    env.close()

if __name__ == "__main__":
    main()