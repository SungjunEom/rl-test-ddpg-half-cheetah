import torch

class RolloutBuffer:
    def __init__(self, capacity, state_dim, action_dim, device):
        self.states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)
        self.logprobs = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.values = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.next_states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.terminated = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.episode_ends = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.capacity = capacity
        self.device = device
        self.ptr = 0

    def push(self, state, action, logprob, reward, value, next_state, terminated, truncated):
        if self.ptr >= self.capacity:
            raise ValueError("Rollout buffer is full")
        self.states[self.ptr].copy_(torch.as_tensor(state, dtype=torch.float32, device=self.device))
        self.actions[self.ptr].copy_(torch.as_tensor(action, dtype=torch.float32, device=self.device))
        self.logprobs[self.ptr].copy_(torch.as_tensor(logprob, dtype=torch.float32, device=self.device))
        self.rewards[self.ptr] = float(reward)
        self.values[self.ptr] = float(value)
        # Preserve the observation returned by step(), before any reset.
        self.next_states[self.ptr].copy_(torch.as_tensor(next_state, dtype=torch.float32, device=self.device))
        self.terminated[self.ptr] = float(terminated)
        self.episode_ends[self.ptr] = float(terminated or truncated)
        self.ptr += 1

    def clear(self):
        self.ptr = 0

    def get(self):
        return (
            self.states[:self.ptr],
            self.actions[:self.ptr],
            self.logprobs[:self.ptr],
            self.rewards[:self.ptr],
            self.values[:self.ptr],
            self.next_states[:self.ptr],
            self.terminated[:self.ptr],
            self.episode_ends[:self.ptr]
        )
