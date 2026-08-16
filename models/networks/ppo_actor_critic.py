import torch
import torch.nn as nn
from torch.distributions import Normal

class PPOActor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(PPOActor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        self.action_log_std = nn.Parameter(torch.zeros(1, action_dim))
        self.max_action = max_action

    def forward(self, state):
        action_mean = self.net(state) * self.max_action
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        dist = Normal(action_mean, action_std)
        return dist

class PPOCritic(nn.Module):
    def __init__(self, state_dim):
        super(PPOCritic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

    def forward(self, state):
        return self.net(state)
