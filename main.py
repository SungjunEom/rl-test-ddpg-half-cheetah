import argparse
import gymnasium as gym
import numpy as np
import torch
import imageio
from torch.utils.tensorboard import SummaryWriter
from models.agent import DDPGAgent
from models.replay_buffer import ReplayBuffer

def save_model(agent, filepath="actor.pth"):
    # Save the trained actor network weights
    torch.save(agent.actor.state_dict(), filepath)
    print(f"\n[Info] Final model saved successfully to: {filepath}")

def evaluate_and_record(agent, env_id="HalfCheetah-v5", filepath="eval_rollout.gif"):
    print(f"\n[Info] Evaluating agent deterministically and recording rollout...")
    # Use rgb_array for recording
    env = gym.make(env_id, render_mode="rgb_array")
    
    raw_frames = []
    state, info = env.reset()
    raw_frames.append(env.render())
    
    episode_reward = 0
    done = False
    
    while not done:
        # Select action deterministically (no noise) for evaluation
        action = agent.select_action(state, noise_scale=0.0)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        
        # Capture raw visual frame
        raw_frames.append(env.render())
        
    print(f"[Info] Evaluation Episode Reward: {episode_reward:.2f}")
    
    # Save frames as an animated GIF
    imageio.mimsave(filepath, raw_frames, fps=30)
    print(f"[Info] Evaluation video saved successfully to: {filepath}")
    env.close()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="DDPG on HalfCheetah-v5")
    parser.add_argument("--eval", action="store_true", help="Only evaluate the saved model instead of training")
    parser.add_argument("--model_path", type=str, default="actor.pth", help="Path to the saved actor model checkpoint")

    # Hyperparameters for tuning
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Actor and Critic optimizers")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for reward decay")
    parser.add_argument("--tau", type=float, default=0.005, help="Soft update target network coefficient")
    parser.add_argument("--batch_size", type=int, default=256, help="Mini-batch size for training updates")
    parser.add_argument("--max_episodes", type=int, default=1000, help="Maximum number of training episodes")
    parser.add_argument("--noise_scale", type=float, default=0.1, help="Standard deviation of exploration Gaussian noise")
    parser.add_argument("--capacity", type=int, default=1000000, help="Replay buffer capacity")
    parser.add_argument("--log_dir", type=str, default="runs/ddpg_halfcheetah", help="TensorBoard log directory")
    args = parser.parse_args()

    env_id = "HalfCheetah-v5"
    env = gym.make(env_id)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = DDPGAgent(
        state_dim=state_dim, 
        action_dim=action_dim, 
        max_action=max_action,
        lr=args.lr,
        gamma=args.gamma,
        tau=args.tau
    )
    
    print(f"[Info] Training on device: {agent.device}")

    if args.eval:
        # Skip training, load saved actor weights, and run evaluation
        try:
            agent.actor.load_state_dict(torch.load(args.model_path, map_location=agent.device))
            print(f"[Info] Successfully loaded model weights from: {args.model_path}")
            evaluate_and_record(agent, env_id=env_id, filepath="eval_rollout.gif")
        except FileNotFoundError:
            print(f"[Error] Could not find saved model weights at: {args.model_path}")
    else:
        # Initialize TensorBoard SummaryWriter
        writer = SummaryWriter(log_dir=args.log_dir)
        print(f"[Info] TensorBoard logging enabled. Runs directory: {args.log_dir}")
        print(f"[Info] To launch TensorBoard, run: tensorboard --logdir={args.log_dir}")

        # Instantiate replay buffer on GPU/CPU
        replay_buffer = ReplayBuffer(capacity=args.capacity, state_dim=state_dim, action_dim=action_dim, device=agent.device)

        max_episodes = args.max_episodes
        batch_size = args.batch_size
        global_step = 0

        for episode in range(max_episodes):
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
                
                if replay_buffer.size > batch_size:
                    critic_loss, actor_loss = agent.train(replay_buffer, batch_size)
                    writer.add_scalar("Loss/Critic", critic_loss, global_step)
                    writer.add_scalar("Loss/Actor", actor_loss, global_step)
                    
            print(f"Episode: {episode + 1}, Reward: {episode_reward:.2f}")
            writer.add_scalar("Reward/Episode", episode_reward, episode + 1)

        # Save final trained weights
        save_model(agent, args.model_path)

        # Record evaluation rollout
        evaluate_and_record(agent, env_id=env_id, filepath="eval_rollout.gif")

        # Close the TensorBoard writer
        writer.close()
    
    env.close()

if __name__ == "__main__":
    main()