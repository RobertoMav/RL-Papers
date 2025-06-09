import random
from collections import deque, namedtuple

import torch

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward", "done"))


class ReplayBuffer:
    def __init__(self, capacity: int, device: torch.device):
        self.memory = deque([], maxlen=capacity)
        self.device = device

    def push(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ):
        """Save a transition"""
        state = state.to(self.device)
        action = action.to(self.device)
        next_state = next_state.to(self.device)
        reward = reward.to(self.device)
        done = done.to(self.device)
        self.memory.append(Transition(state, action, next_state, reward, done))

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)
