import torch
import torch.nn as nn

class PixelEncoder(nn.Module):
    def __init__(self, num_stacked_frames=4, feature_dim=50):
        super().__init__()
        # Input shape: (Batch, num_stacked_frames, 84, 84)
        self.conv = nn.Sequential(
            nn.Conv2d(num_stacked_frames, 32, kernel_size=8, stride=4), # 20x20
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # 9x9
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # 7x7
            nn.ReLU(),
            nn.Flatten()
        )
        # Linear layer to project features to a compact embedding
        self.fc = nn.Linear(64 * 7 * 7, feature_dim)
        self.ln = nn.LayerNorm(feature_dim)
        
    def forward(self, x):
        x = x / 255.0 # Normalize pixels on the fly
        x = self.conv(x)
        x = torch.relu(self.ln(self.fc(x)))
        return x