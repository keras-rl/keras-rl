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
                 gamma=.99, batch_size=6, nb_steps_warmup=0, train_interval=1, memory_interval=1 ):


        super(CEMAgent, self).__init__()

        # Parameters.
        self.nb_actions = nb_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.nb_steps_warmup = nb_steps_warmup
        self.train_interval = train_interval
        self.memory_interval = memory_interval
        self.window_length = window_length

        self.theta = model.get_params()
        self.episode=0


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
        pass#self.model.save_weights(filepath, overwrite=overwrite)

    def reset_states(self):
        self.recent_action = None
        self.recent_observations = deque(maxlen=self.window_length)
        self.recent_params = deque(maxlen=self.window_length)

    def select_action(self,state):
        batch = state.copy()
        return self.model.select_action(batch)

    def update_theta(self,theta):
        self.theta = theta

    def choose_weights(self):
        cov = np.diag(np.ones(self.theta.shape[0]))
        params = np.random.multivariate_normal(self.theta,cov)
        self.model.update_params(params)

    def forward(self, observation):
        # Select an action.
        while len(self.recent_observations) < self.recent_observations.maxlen:
            # Not enough data, fill the recent_observations queue with copies of the current input.
            # This allows us to immediately perform a policy action instead of falling back to random
            # actions.
            self.recent_observations.append(np.copy(observation))
        state = np.array(list(self.recent_observations)[1:] + [observation])
        # Select an action.
        action = self.select_action(state)

        # Book-keeping.
        self.recent_observations.append(observation)
        self.recent_action = action
        self.recent_params.append(self.model.get_params())


        return action

    def backward(self, reward, terminal):
        metrics = [np.nan for _ in ['mse']]
        if not self.training:
            # We're done here. No need to update the experience memory since we only use the working
            # memory to obtain the state over the most recent observations.
            return metrics


        # Store most recent experience in memory.

        self.memory.append(reward)

        if (terminal):

            self.memory.finalise_episode(self.recent_params[-1])
            self.episode += 1

            if (self.episode % self.batch_size == 0):
                params , reward_totals = self.memory.sample(self.batch_size)
                best_idx = np.argsort(np.array(reward_totals))
                new_params = np.mean(np.vstack(params[i] for i in best_idx),axis=0)
                assert new_params.shape == params[0].shape
                self.update_theta(new_params)
                print("UPDATED MODEL , NEW PARAMS")
                print(new_params)

            self.choose_weights()
        return metrics

        # Train the network on a single stochastic batch.
        # if self.step > self.nb_steps_warmup and self.step % self.train_interval == 0:
        #
        #     params , experiences = self.memory.sample(self.batch_size, self.window_length)
        #
        #     assert len(experiences) == self.batch_size == 32 == len(params)
        #     best_idx = np.argsort(rewards)[:10]
        #
        #     new_params = np.mean(np.vstack(params[i] for i in best_idx),axis=0)
        #     print("new mean")
        #     print(new_params)
        #     assert new_params.shape == params[0].shape
        #     #print("new params")
            #print(new_params)

        #    self.update_model(new_params)

        ## get the rewards for a stochastic sample



        #    metrics += [np.nan]

        return metrics

    @property
    def metrics_names(self):
        return self.model.metrics_names[:]
