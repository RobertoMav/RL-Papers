from collections import deque

import ale_py
import gymnasium as gym
import numpy as np
from gymnasium import spaces

gym.register_envs(ale_py)


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(obs_shape[0], obs_shape[1], 1), dtype=np.uint8
        )

    def observation(self, obs):
        import cv2

        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        return obs[:, :, np.newaxis]


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        self.shape = (shape, shape)
        obs_shape = self.shape + (self.observation_space.shape[2],)
        self.observation_space = spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, obs):
        import cv2

        return cv2.resize(obs, self.shape, interpolation=cv2.INTER_AREA)


class FrameStack(gym.ObservationWrapper):
    def __init__(self, env, k):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(k, shp[0], shp[1]),
            dtype=env.observation_space.dtype,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        assert len(self.frames) == self.k
        # Assuming the frames are (H, W, 1), we squeeze the last dim
        # and stack along the first axis to get (k, H, W)
        return np.stack([np.squeeze(f) for f in self.frames], axis=0)

    def observation(self, observation):
        # This method is not used in the stacked-frames approach.
        pass


def create_env(env_name, shape=84, k=4):
    env = gym.make(env_name)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape)
    env = FrameStack(env, k)
    return env
