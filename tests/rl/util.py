import numpy as np
import random

from rl.core import Env


class MultiInputTestEnv(Env):
    def __init__(self, observation_shape):
        self.observation_shape = observation_shape

    def step(self, action):
        return self._get_obs(), random.choice([0, 1]), random.choice([True, False]), {}

    def reset(self):
        return self._get_obs()

    def _get_obs(self):
        if type(self.observation_shape) is list:
            return [np.random.random(s) for s in self.observation_shape]
        else:
            return np.random.random(self.observation_shape)
