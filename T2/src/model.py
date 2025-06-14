import torch
from torch import nn


class DQN(nn.Module):
    def __init__(self, n_stack_frames: int, n_actions: int):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(n_stack_frames, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            # Input: 84x84, Conv1: (84-5)/2+1=40, Conv2: (40-3)/2+1=19, Conv3: (19-3)/1+1=17
            # Final size: 64 * 17 * 17 = 18496
            nn.Linear(18496, 512),
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
