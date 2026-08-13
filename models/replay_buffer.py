import random
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity, state_shape, action_dim, device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # Pre-allocate memory directly on GPU for zero CPU-GPU copy overhead during training
        self.states = torch.zeros((capacity, *state_shape), dtype=torch.uint8, device=device)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.next_states = torch.zeros((capacity, *state_shape), dtype=torch.uint8, device=device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32, device=device)

    def push(self, state, action, reward, next_state, done):
        # Input state and next_state are already GPU tensors
        self.states[self.ptr].copy_(state)
        self.actions[self.ptr].copy_(torch.as_tensor(action, device=self.device))
        self.rewards[self.ptr] = float(reward)
        self.next_states[self.ptr].copy_(next_state)
        self.dones[self.ptr] = float(done)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        ind = torch.randint(0, self.size, (batch_size,), device=self.device)
        # Convert uint8 to float32 on-the-fly on GPU
        return (
            self.states[ind].float(),
            self.actions[ind],
            self.rewards[ind],
            self.next_states[ind].float(),
            self.dones[ind]
        )