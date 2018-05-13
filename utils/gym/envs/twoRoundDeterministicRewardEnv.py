import gym
from ..spaces import Discrete


class TwoRoundDeterministicRewardEnv(gym.Env):
    def __init__(self):
        self.action_space = Discrete(2)
        self.observation_space = Discrete(3)
        self.reset()

    def step(self, action):
        rewards = [[0, 3], [1, 2]]

        assert self.action_space.contains(action)

        if self.firstAction is None:
            self.firstAction = action
            reward = 0
            done = False
        else:
            reward = rewards[self.firstAction][action]
            done = True

        return self.get_obs(), reward, done, {}

    def get_obs(self):
        if self.firstAction is None:
            return 2
        else:
            return self.firstAction

    def reset(self):
        self.firstAction = None
        return self.get_obs()
