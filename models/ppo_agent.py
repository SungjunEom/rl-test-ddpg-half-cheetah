import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models.networks.ppo_actor_critic import PPOActor, PPOCritic


@torch.no_grad()
def compute_gae(rewards, values, next_values, terminated, episode_ends, gamma, gae_lambda):
    """Bootstrap timeouts, but never propagate advantages across resets."""
    deltas = rewards + gamma * (1.0 - terminated) * next_values - values
    # Run the sequential recurrence on CPU to avoid thousands of tiny GPU kernels.
    delta_cpu = deltas.cpu().numpy()
    continuation = (1.0 - episode_ends).cpu().numpy()
    advantages = np.empty_like(delta_cpu)
    gae = np.zeros_like(delta_cpu[0])
    for t in reversed(range(len(rewards))):
        gae = delta_cpu[t] + gamma * gae_lambda * continuation[t] * gae
        advantages[t] = gae
    advantages = torch.as_tensor(advantages, device=rewards.device)
    return advantages, advantages + values


class PPOAgent:
    def __init__(self, state_dim, action_dim, max_action, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_ratio=0.2, ppo_epochs=10, device="cuda"):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.ppo_epochs = ppo_epochs
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_action = max_action

        self.actor = PPOActor(state_dim, action_dim, max_action).to(self.device)
        self.critic = PPOCritic(state_dim).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist = self.actor(state)
            action = dist.sample()
            logprob = dist.log_prob(action)
            value = self.critic(state)
            
        action = action.cpu().data.numpy().flatten()
        return action, logprob.cpu().data.numpy().flatten(), value.item()

    def evaluate(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_mean = self.actor.net(state) * self.actor.max_action
        return action_mean.cpu().data.numpy().flatten()

    def train(self, buffer):
        states, actions, old_logprobs, rewards, values, next_states, terminated, episode_ends = buffer.get()
        
        # Calculate advantages and returns using GAE
        with torch.no_grad():
            next_values = self.critic(next_states)
            advantages, returns = compute_gae(
                rewards, values, next_values, terminated, episode_ends,
                self.gamma, self.gae_lambda,
            )
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        old_logprobs = old_logprobs.sum(dim=-1, keepdim=True)
        
        actor_losses = []
        critic_losses = []

        for _ in range(self.ppo_epochs):
            dist = self.actor(states)
            logprobs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
            dist_entropy = dist.entropy().sum(dim=-1, keepdim=True)
            state_values = self.critic(states)

            ratios = torch.exp(logprobs - old_logprobs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * dist_entropy.mean()

            critic_loss = nn.MSELoss()(state_values, returns)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())

        buffer.clear()
        
        return np.mean(critic_losses), np.mean(actor_losses)
