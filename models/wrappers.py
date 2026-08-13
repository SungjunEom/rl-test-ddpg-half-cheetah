import gymnasium as gym
import numpy as np
import torch

class PendulumPixelWrapper(gym.Wrapper):
    def __init__(self, env, device, num_stacked_frames=4, size=(84, 84)):
        super().__init__(env)
        self.device = device
        self.num_stacked_frames = num_stacked_frames
        self.size = size
        self.frames = []
        
        # Define the new observation space (4, 84, 84)
        self.observation_space = gym.spaces.Box(
            low=0, 
            high=255, 
            shape=(num_stacked_frames, size[0], size[1]), 
            dtype=np.uint8
        )

    def _get_pixel_obs(self):
        # Render the environment as an RGB array
        frame = self.env.render()  # Returns (500, 500, 3)
        # Transfer raw frame to GPU immediately for fast operations
        frame_t = torch.from_numpy(frame).to(self.device, non_blocking=True)
        # Grayscale on GPU (Y = 0.299R + 0.587G + 0.114B)
        frame_t = frame_t.permute(2, 0, 1).float()
        gray_t = 0.299 * frame_t[0] + 0.587 * frame_t[1] + 0.114 * frame_t[2]
        # Resize on GPU using bilinear interpolation
        gray_t = gray_t.unsqueeze(0).unsqueeze(0)
        resized_t = torch.nn.functional.interpolate(gray_t, size=self.size, mode='bilinear', align_corners=False)
        return resized_t.squeeze(0).squeeze(0).byte()  # Shape: (84, 84), dtype: uint8 on GPU

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        pixel_obs = self._get_pixel_obs()
        self.frames = [pixel_obs] * self.num_stacked_frames
        return torch.stack(self.frames, dim=0), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        pixel_obs = self._get_pixel_obs()
        self.frames.pop(0)
        self.frames.append(pixel_obs)
        return torch.stack(self.frames, dim=0), reward, terminated, truncated, info
