import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import scipy.stats

"""
You can generate the final plot by simply running:
    conda run -n aloc python compare_runs.py
"""

def get_scalars(log_dir, tag):
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

def get_all_runs_data(algo, num_runs=20, tag="Reward/Episode"):
    all_values = []
    min_len = float('inf')
    
    # First pass to find minimum length
    runs_data = []
    for i in range(1, num_runs + 1):
        log_dir = f"runs/{algo}_halfcheetah_run_{i}"
        steps, values = get_scalars(log_dir, tag)
        if len(values) > 0:
            runs_data.append(values)
            min_len = min(min_len, len(values))
    
    if len(runs_data) == 0:
        return None, None, None, None

    # Truncate all runs to min_len to easily calculate mean and std
    truncated_data = [run[:min_len] for run in runs_data]
    data_array = np.array(truncated_data)
    
    mean = np.mean(data_array, axis=0)
    std = np.std(data_array, axis=0)
    
    # Calculate 95% confidence interval
    n = data_array.shape[0]
    ci = 1.96 * std / np.sqrt(n)
    
    steps = np.arange(1, min_len + 1)
    return steps, mean, ci, data_array

def main():
    num_runs = 20
    
    ddpg_steps, ddpg_mean, ddpg_ci, ddpg_data = get_all_runs_data("ddpg", num_runs)
    ppo_steps, ppo_mean, ppo_ci, ppo_data = get_all_runs_data("ppo", num_runs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Learning Curve
    if ddpg_steps is not None:
        ax1.plot(ddpg_steps, ddpg_mean, label=f"DDPG", color='blue')
        ax1.fill_between(ddpg_steps, ddpg_mean - ddpg_ci, ddpg_mean + ddpg_ci, color='blue', alpha=0.2)
        
    if ppo_steps is not None:
        ax1.plot(ppo_steps, ppo_mean, label=f"PPO", color='orange')
        ax1.fill_between(ppo_steps, ppo_mean - ppo_ci, ppo_mean + ppo_ci, color='orange', alpha=0.2)
        
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.set_title(f"Learning Curve on HalfCheetah ({num_runs} Runs)")
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Final Scores Bar Chart
    algos = []
    means = []
    cis = []
    colors = []
    
    if ddpg_data is not None:
        # Final scores are the mean of the last 10 episodes to reduce noise
        final_scores_ddpg = np.mean(ddpg_data[:, -10:], axis=1)
        algos.append("DDPG")
        means.append(np.mean(final_scores_ddpg))
        cis.append(1.96 * np.std(final_scores_ddpg) / np.sqrt(num_runs))
        colors.append('blue')
        
    if ppo_data is not None:
        final_scores_ppo = np.mean(ppo_data[:, -10:], axis=1)
        algos.append("PPO")
        means.append(np.mean(final_scores_ppo))
        cis.append(1.96 * np.std(final_scores_ppo) / np.sqrt(num_runs))
        colors.append('orange')
        
    if algos:
        ax2.bar(algos, means, yerr=cis, capsize=10, color=colors, alpha=0.7)
        ax2.set_ylabel("Final Score (Last 10 Episodes Mean)")
        ax2.set_title("Final Scores Comparison")
        ax2.grid(True, axis='y')

    plt.tight_layout()
    plt.savefig("comparison_plot.png")
    print("Plot successfully saved to comparison_plot.png")

if __name__ == "__main__":
    main()
