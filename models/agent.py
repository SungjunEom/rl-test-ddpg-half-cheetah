import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models.networks.actor_critic import Actor, Critic

class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action, lr=1e-3, gamma=0.99, tau=0.005, device="cuda"):
        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Main networks
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        
        # Target networks
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        
        # Hard copy weights initially
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

    def select_action(self, state, noise_scale=0.1):
        # state is expected to be a numpy array
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        
        # Add Gaussian noise for exploration
        noise = np.random.normal(0, noise_scale, size=action.shape)
        return np.clip(action + noise, -self.max_action, self.max_action)

    def train(self, replay_buffer, batch_size=128):
        # 1. Sample from buffer
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        # 2. Update Critic
        with torch.no_grad():
            target_actions = self.actor_target(next_states)
            target_Q = self.critic_target(next_states, target_actions)
            target_Q = rewards + ((1 - dones) * self.gamma * target_Q)

        current_Q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_Q, target_Q)

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        # 3. Update Actor (Maximize Critic output)
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        # 4. Soft update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        return critic_loss.item(), actor_loss.item()