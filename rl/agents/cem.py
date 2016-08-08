from __future__ import division
from collections import deque
from copy import deepcopy

import numpy as np
import keras.backend as K
from keras.models import Model

from rl.core import Agent
from rl.util import *


class CEMAgent(Agent):
    def __init__(self, model, nb_actions, memory, window_length=1,
                 gamma=.99, batch_size=32, nb_steps_warmup=0, train_interval=1, memory_interval=1 ):


        super(CEMAgent, self).__init__()

        # Parameters.
        self.nb_actions = nb_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.nb_steps_warmup = nb_steps_warmup
        self.train_interval = train_interval
        self.memory_interval = memory_interval
        self.window_length = window_length


        # Related objects.
        self.model = model
        self.memory = memory

        # State.
        self.compiled = False
        self.reset_states()

    def compile(self, optimizer, metrics=[]):
        self.compiled = True


    def load_weights(self, filepath):
        self.model.load_weights(filepath)
        self.update_target_model_hard()

    def save_weights(self, filepath, overwrite=False):
        self.model.save_weights(filepath, overwrite=overwrite)

    def reset_states(self):
        self.recent_action = None
        self.recent_observations = deque(maxlen=self.window_length)

    def select_action(self):
        return self.model.select_action()

    def forward(self, observation):

        # Select an action.
        action = self.select_action()

        # Book-keeping.
        self.recent_action = action

        return action

    def backward(self, reward, terminal):
        metrics = [np.nan for _ in ['mse']]
        if not self.training:
            # We're done here. No need to update the experience memory since we only use the working
            # memory to obtain the state over the most recent observations.
            return metrics



        # Store most recent experience in memory.
        if self.step % self.memory_interval == 0:
            self.memory.append(self.recent_observations[-1], self.recent_action, reward, terminal)

        # Train the network on a single stochastic batch.
        if self.step > self.nb_steps_warmup and self.step % self.train_interval == 0:
            idx=np.random.sample(self.step,self.train_interval)

            params , experiences = self.memory.sample(self.batch_size, self.window_length)
            assert len(experiences) == self.batch_size

        ## get the rewards for a stochastic sample



        #    metrics += [np.nan]

        return metrics

    @property
    def metrics_names(self):
        return self.model.metrics_names[:]
