import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def get_scalars(log_dir, tag):
    # Find the event file in the directory
    if not os.path.exists(log_dir):
        return [], []
        
    ea = event_accumulator.EventAccumulator(log_dir, size_guidance={'scalars': 0})
    ea.Reload()
    
    if tag in ea.Tags().get('scalars', []):
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        return steps, values
    return [], []

def main():
    ddpg_log = "runs/ddpg_halfcheetah"
    ppo_log = "runs/ppo_halfcheetah"

    ddpg_steps, ddpg_rewards = get_scalars(ddpg_log, "Reward/Episode")
    ppo_steps, ppo_rewards = get_scalars(ppo_log, "Reward/Episode")

    plt.figure(figsize=(10, 6))
    if ddpg_steps:
        plt.plot(ddpg_steps, ddpg_rewards, label="DDPG", alpha=0.8, linewidth=2)
    if ppo_steps:
        plt.plot(ppo_steps, ppo_rewards, label="PPO", alpha=0.8, linewidth=2)
        
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("DDPG vs PPO on HalfCheetah (100 Episodes)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("comparison_plot.png")
    print("Plot successfully saved to comparison_plot.png")

if __name__ == "__main__":
    main()
