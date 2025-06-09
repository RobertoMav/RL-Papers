import torch
from torch import nn


class DQN(nn.Module):
    def __init__(self, n_stack_frames: int, n_actions: int):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(n_stack_frames, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The input from our wrapper is already in the format (N, C, H, W)
        # where C is the number of stacked frames.
        if len(x.shape) == 3:  # If we get a single observation (C, H, W)
            x = x.unsqueeze(0)
        # Normalize pixel values
        return self.network(x / 255.0)
