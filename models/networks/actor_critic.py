import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, encoder, action_dim, max_action):
        super(Actor, self).__init__()
        self.encoder = encoder
        # Use the out_features of the encoder linear layer to determine features dimension
        feature_dim = encoder.fc.out_features
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh() # Scales output to [-1, 1]
        )
        self.max_action = max_action

    def forward(self, state, detach_encoder=False):
        if detach_encoder:
            features = self.encoder(state).detach()
        else:
            features = self.encoder(state)
        return self.max_action * self.net(features)

class Critic(nn.Module):
    def __init__(self, encoder, action_dim):
        super(Critic, self).__init__()
        self.encoder = encoder
        feature_dim = encoder.fc.out_features
        # Concat state features and action at the input layer
        self.net = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        features = self.encoder(state)
        sa = torch.cat([features, action], 1)
        return self.net(sa)