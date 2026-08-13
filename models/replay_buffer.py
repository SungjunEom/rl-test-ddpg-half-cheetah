import random
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim, device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # Pre-allocate memory directly on GPU for zero CPU-GPU copy overhead during training
        self.states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.next_states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32, device=device)

    def push(self, state, action, reward, next_state, done):
        self.states[self.ptr].copy_(torch.as_tensor(state, device=self.device, dtype=torch.float32))
        self.actions[self.ptr].copy_(torch.as_tensor(action, device=self.device, dtype=torch.float32))
        self.rewards[self.ptr] = float(reward)
        self.next_states[self.ptr].copy_(torch.as_tensor(next_state, device=self.device, dtype=torch.float32))
        self.dones[self.ptr] = float(done)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        ind = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.states[ind],
            self.actions[ind],
            self.rewards[ind],
            self.next_states[ind],
            self.dones[ind]
        )