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
                 gamma=.99, batch_size=50, nb_steps_warmup=1000, train_interval=50, elite_frac=0.05, memory_interval=1):

        super(CEMAgent, self).__init__()

        # Parameters.
        self.nb_actions = nb_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.elite_frac = elite_frac
        self.num_best = int(self.batch_size*self.elite_frac)
        self.nb_steps_warmup = nb_steps_warmup
        self.train_interval = train_interval
        self.memory_interval = memory_interval
        self.window_length = window_length
        self.episode=0

        # Related objects.
        self.memory = memory

        self.model = model
        self.shapes = [w.shape for w in model.get_weights()]
        self.sizes = [w.size for w in model.get_weights()]
        self.num_weights = sum(self.sizes)

        self.update_theta(None)

        # State.
        self.compiled = False
        self.reset_states()

    def compile(self):
        self.model.compile(optimizer='sgd', loss='mse')
        self.compiled = True

    def load_weights(self, filepath):
        self.model.load_weights(filepath)
        self.update_target_model_hard()

    def save_weights(self, filepath, overwrite=False):
        self.model.save_weights(filepath, overwrite=overwrite)

    def get_flat_weights(self):
        weights_flat = np.zeros(self.num_weights)
        weights = self.model.get_weights()
        pos=0
        for i_layer, size in enumerate(self.sizes):
            weights_flat[pos:pos+size] = weights[i_layer].flatten()
            pos += size

        return weights_flat

    def reset_states(self):
        self.recent_action = None
        self.recent_observations = deque(maxlen=self.window_length)
        self.recent_params = deque(maxlen=self.window_length)

    def select_action(self,state,stochastic=True):
        batch = state.copy()
        action = self.model.predict_on_batch(batch).flatten()
        if (stochastic):
            return np.random.choice(np.arange(self.nb_actions),p=action/np.sum(action))
        return np.argmax(action)

    def update_theta(self,theta):
        if (theta is not None):
            self.theta = theta
        else:
            # theta = (means,covariances)
            self.theta = np.hstack((np.random.randn(self.num_weights),np.random.randn(self.num_weights)))

    def choose_weights(self):

        mean = self.theta[:self.num_weights]
        cov = self.theta[self.num_weights:]
        weights_flat = np.square(cov) * np.random.randn(self.num_weights) + mean

        sampled_weights = []
        pos=0
        for i_layer, size in enumerate(self.sizes):
            arr = weights_flat[pos:pos+size].reshape(self.shapes[i_layer])
            sampled_weights.append(arr)
            pos += size

        self.model.set_weights(sampled_weights)

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
        self.recent_params.append(self.get_flat_weights())

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

            if (self.step > self.nb_steps_warmup and self.episode % self.train_interval == 0):
                params, reward_totals = self.memory.sample(self.batch_size)
                best_idx = np.argsort(np.array(reward_totals))[-self.num_best:]
                best = np.vstack(params[i] for i in best_idx)

                assert (not np.isnan(best).any())
                mean = np.mean(best,axis=0)
                print("new mean")
                print(mean)
                print("rewards")
                print([reward_totals[i] for i in best_idx])
                cov = np.std(best,axis=0)
                print("new cov")
                print(cov)
                new_params = np.hstack((mean,cov))
                assert new_params.shape == self.theta.shape
                self.update_theta(new_params)

            self.choose_weights()

        return metrics

    @property
    def metrics_names(self):
        return self.model.metrics_names[:]
