from rl.core import Agent
from rl.util import *


class RandomAgent(Agent):
    def __init__(self, nb_actions):
        self.nb_actions = nb_actions
        self.compiled = False
        super(RandomAgent, self).__init__()

    def forward(self, observation):
        return np.random.randint(self.nb_actions)

    def backward(self, reward, terminal):
        pass

    def compile(self, optimizer, metrics=[]):
        self.compiled = True

    def load_weights(self, filepath):
        pass

    def save_weights(self, filepath, overwrite=False):
        pass

    @property
    def layers(self):
        return None
