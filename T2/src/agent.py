import random
from copy import deepcopy

import numpy as np
import torch
from torch import optim

from src.model import DQN
from src.replay_buffer import ReplayBuffer, Transition


class Agent:
    def __init__(
        self,
        n_stack_frames: int,
        n_actions: int,
        device: torch.device,
        lr: float,
        gamma: float,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay: float,
        batch_size: int,
        replay_buffer_size: int,
        target_update_frequency: int,
    ):
        self.n_actions = n_actions
        self.device = device
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency

        self.policy_net = DQN(n_stack_frames, n_actions).to(device)
        self.target_net = deepcopy(self.policy_net)
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(replay_buffer_size, device)
        self.steps_done = 0

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        sample = random.random()
        self.epsilon = self.epsilon_end + (self.epsilon - self.epsilon_end) * np.exp(
            -1.0 * self.steps_done / self.epsilon_decay
        )
        self.steps_done += 1
        if sample > self.epsilon:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor(
                [[random.randrange(self.n_actions)]],
                device=self.device,
                dtype=torch.long,
            )

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        done_batch = torch.cat(batch.done)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        non_final_mask = ~done_batch
        next_state_values[non_final_mask] = (
            self.target_net(next_state_batch[non_final_mask]).max(1)[0].detach()
        )

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        if self.steps_done % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
