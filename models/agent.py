import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models.networks.actor_critic import Actor, Critic
from models.networks.visual_encoder import PixelEncoder

class DDPGAgent:
    def __init__(self, num_stacked_frames, action_dim, max_action, lr=1e-3, gamma=0.99, tau=0.005, device="cuda"):
        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Create single shared visual encoder and its target counterpart
        self.encoder = PixelEncoder(num_stacked_frames).to(self.device)
        self.encoder_target = PixelEncoder(num_stacked_frames).to(self.device)
        self.encoder_target.load_state_dict(self.encoder.state_dict())

        # Main networks (sharing the same encoder)
        self.actor = Actor(self.encoder, action_dim, max_action).to(self.device)
        self.critic = Critic(self.encoder, action_dim).to(self.device)
        
        # Target networks (sharing the same target encoder)
        self.actor_target = Actor(self.encoder_target, action_dim, max_action).to(self.device)
        self.critic_target = Critic(self.encoder_target, action_dim).to(self.device)
        
        # Hard copy weights initially
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        # Critic optimizer updates both Critic MLP and Shared Encoder
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Actor optimizer updates ONLY Actor MLP parameters (not the shared encoder)
        actor_mlp_params = [p for name, p in self.actor.named_parameters() if "encoder" not in name]
        self.actor_optimizer = optim.Adam(actor_mlp_params, lr=lr)

    def select_action(self, state, noise_scale=0.1):
        # state is a GPU uint8 tensor; convert to float and add batch dimension
        state = state.float().unsqueeze(0)
        action = self.actor(state).cpu().data.numpy().flatten()
        
        # Add Gaussian noise for exploration
        noise = np.random.normal(0, noise_scale, size=action.shape)
        return np.clip(action + noise, -self.max_action, self.max_action)

    def train(self, replay_buffer, batch_size=128):
        # 1. Sample from buffer (already on GPU, states cast to float)
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        # 2. Update Critic (which updates the shared CNN)
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
        # We freeze critic gradients while updating the actor
        # detach_encoder=True ensures actor updates do not propagate gradients back to the shared CNN
        actor_loss = -self.critic(states, self.actor(states, detach_encoder=True)).mean()
        
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        # 4. Soft update target networks (each param updated exactly once)
        # Updates shared encoder and critic MLP parameters
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        # Updates Actor MLP parameters (excluding the shared encoder)
        actor_mlp_params = [p for name, p in self.actor.named_parameters() if "encoder" not in name]
        actor_target_mlp_params = [p for name, p in self.actor_target.named_parameters() if "encoder" not in name]
        for param, target_param in zip(actor_mlp_params, actor_target_mlp_params):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        return critic_loss.item(), actor_loss.item()