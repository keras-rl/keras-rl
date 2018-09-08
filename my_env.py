from random import randint

import numpy as np
import sys
from six import StringIO, b

import gym
from gym import utils, spaces
from gym.envs.toy_text import discrete
from gym.utils import seeding

DOWN = 0
RIGHT = 1
UP = 2
LEFT = 3

ACTIONS = [UP, RIGHT, DOWN, LEFT]

NUM_ROWS = 4
NUM_COLS = 4

NUM_ACTIONS = len(ACTIONS)
NUM_STATES = NUM_ROWS * NUM_COLS


class BlockWorldEnvironmentConv(gym.Env):
    def __init__(self, height, width, initial_x, initial_y, final_x, final_y):
        self.height = height
        self.width = width
        self.initial_x = initial_x
        self.initial_y = initial_y
        self.my_x = initial_x
        self.my_y = initial_y
        self.goal_x = final_x
        self.goal_y = final_y
        self._DOWN = 1
        self._UP = 0
        self._RIGHT = 3
        self._LEFT = 2
        self.steps = 0
        self.actions = [self._UP, self._DOWN, self._LEFT, self._RIGHT]
        low = np.array([0 for i in range(64)])
        high = np.array([2 for i in range(64)])
        self.observation_space = spaces.Box(low, high)
        self.action_space = spaces.Discrete(4)
        self.reward = 0
        self.zero = np.zeros((64,1))
        self.done = False

    def reset(self):
        self.my_x = self.initial_x  # randint(0, self.width-1)
        self.my_y = self.initial_y  # randint(0, self.height-1)
        self.actions = [self._UP, self._DOWN, self._LEFT, self._RIGHT]
        self.reward = 0
        self.done = False
        self.steps = 0

        return np.array(self.get_state()).reshape((64,1))

    def step(self, action):
        temp_reward = 0
        self.steps += 1

        if self.my_x == self.goal_x and self.my_y == self.goal_y:
            self.done = True
            self.reward += 10
            temp_reward = 10
            return np.array(self.get_state()).reshape((64,1)), 10, self.done, {}
        elif self.my_x == 0 and action == self._LEFT:
            self.reward -= 1
            temp_reward = -1
        elif self.my_x == self.width - 1 and action == self._RIGHT:
            self.reward -= 1
            temp_reward = -1
        elif self.my_y == 0 and action == self._DOWN:
            self.reward -= 1
            temp_reward = -1
        elif self.my_y == self.height - 1 and action == self._UP:
            self.reward -= 1
            temp_reward = -1
        else:
            if action == self._UP:
                self.my_y += 1
                self.reward -= 1
                temp_reward = -1
            elif action == self._DOWN:
                self.my_y -= 1
                self.reward -= 1
                temp_reward = -1
            elif action == self._LEFT:
                self.my_x -= 1
                self.reward -= 1
                temp_reward = -1
            elif action == self._RIGHT:
                self.my_x += 1
                self.reward -= 1
                temp_reward = -1

        return np.array(self.get_state()).reshape((64,1)), temp_reward, self.done, {}

    def get_state(self):
        if not self.done:
            environment = [
                [0 for j in range(self.width)] for i in range(self.height)
            ]
            environment[-self.my_y][self.my_x] = 1
            environment[-self.goal_y][self.goal_x] = 2
            return self._flatten(environment)
        else:
            return [0 for i in range(self.height*self.width)]

    def _flatten(self, environment):
        flattened = []
        for i in range(len(environment)):
            for j in range(len(environment[i])):
                flattened.append(environment[i][j])
        return flattened

    def render(self, mode='human'):
        for i in reversed(range(self.height)):
            for j in range(self.width):
                if i == self.my_y and j == self.my_x:
                    print('X', end='')
                elif i == self.goal_x and j == self.goal_y:
                    print('G', end='')
                else:
                    print('0', end='')
            print('')
        print('')


class BlockWorldEnvironmentNonConv(gym.Env):
    def __init__(self, height, width, initial_x, initial_y, final_x, final_y):
        self.height = height
        self.width = width
        self.initial_x = initial_x
        self.initial_y = initial_y
        self.my_x = initial_x
        self.my_y = initial_y
        self.goal_x = final_x
        self.goal_y = final_y
        self._DOWN = 1
        self._UP = 0
        self._RIGHT = 3
        self._LEFT = 2
        self.steps = 0
        self.actions = [self._UP, self._DOWN, self._LEFT, self._RIGHT]
        low = np.array([0 for i in range(64)])
        high = np.array([2 for i in range(64)])
        self.observation_space = spaces.Box(low, high)
        self.action_space = spaces.Discrete(4)
        self.reward = 0
        self.zero = np.zeros((64,1))
        self.done = False

    def reset(self):
        self.my_x = self.initial_x  # randint(0, self.width-1)
        self.my_y = self.initial_y  # randint(0, self.height-1)
        self.actions = [self._UP, self._DOWN, self._LEFT, self._RIGHT]
        self.reward = 0
        self.done = False
        self.steps = 0

        return np.array(self.get_state())

    def step(self, action):
        temp_reward = 0
        self.steps += 1

        if self.my_x == self.goal_x and self.my_y == self.goal_y:
            self.done = True
            self.reward += 10
            temp_reward = 10
            return np.array(self.get_state()), 10, self.done, {}
        elif self.my_x == 0 and action == self._LEFT:
            self.reward -= 1
            temp_reward = -1
        elif self.my_x == self.width - 1 and action == self._RIGHT:
            self.reward -= 1
            temp_reward = -1
        elif self.my_y == 0 and action == self._DOWN:
            self.reward -= 1
            temp_reward = -1
        elif self.my_y == self.height - 1 and action == self._UP:
            self.reward -= 1
            temp_reward = -1
        else:
            if action == self._UP:
                self.my_y += 1
                self.reward -= 1
                temp_reward = -1
            elif action == self._DOWN:
                self.my_y -= 1
                self.reward -= 1
                temp_reward = -1
            elif action == self._LEFT:
                self.my_x -= 1
                self.reward -= 1
                temp_reward = -1
            elif action == self._RIGHT:
                self.my_x += 1
                self.reward -= 1
                temp_reward = -1

        return np.array(self.get_state()), temp_reward, self.done, {}

    def get_state(self):
        if not self.done:
            environment = [
                [0 for j in range(self.width)] for i in range(self.height)
            ]
            environment[-self.my_y][self.my_x] = 1
            environment[-self.goal_y][self.goal_x] = 2
            return self._flatten(environment)
        else:
            return [0 for i in range(self.height*self.width)]

    def _flatten(self, environment):
        flattened = []
        for i in range(len(environment)):
            for j in range(len(environment[i])):
                flattened.append(environment[i][j])
        return flattened

    def render(self, mode='human'):
        for i in reversed(range(self.height)):
            for j in range(self.width):
                if i == self.my_y and j == self.my_x:
                    print('X', end='')
                elif i == self.goal_x and j == self.goal_y:
                    print('G', end='')
                else:
                    print('0', end='')
            print('')
        print('')
