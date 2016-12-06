from __future__ import division
from collections import deque
from copy import deepcopy

import numpy as np
import keras.backend as K
from keras.models import Model

from rl.core import Agent
from rl.util import *

class CEMAgent(Agent):
    def __init__(self, model, nb_actions, memory, batch_size=50, nb_steps_warmup=1000,
                 train_interval=50, elite_frac=0.05, memory_interval=1, theta_init=None,
                 noise_decay_const=0.0, noise_ampl=0.0, **kwargs):
        super(CEMAgent, self).__init__(**kwargs)

        # Parameters.
        self.nb_actions = nb_actions
        self.batch_size = batch_size
        self.elite_frac = elite_frac
        self.num_best = int(self.batch_size * self.elite_frac)
        self.nb_steps_warmup = nb_steps_warmup
        self.train_interval = train_interval
        self.memory_interval = memory_interval
        
        # if using noisy CEM, the minimum standard deviation will be ampl * exp (- decay_const * step )
        self.noise_decay_const = noise_decay_const
        self.noise_ampl = noise_ampl
                
        # default initial mean & cov, override this by passing an theta_init argument
        self.init_mean = 0.0
        self.init_stdev = 1.0
        
        # Related objects.
        self.memory = memory
        self.model = model
        self.shapes = [w.shape for w in model.get_weights()]
        self.sizes = [w.size for w in model.get_weights()]
        self.num_weights = sum(self.sizes)
        
        # store the best result seen during training, as a tuple (reward, flat_weights)
        self.best_seen = (-np.inf, np.zeros(self.num_weights))

        self.theta = np.zeros(self.num_weights*2)
        self.update_theta(theta_init)

        # State.
        self.episode = 0
        self.compiled = False
        self.reset_states()

    def compile(self):
        self.model.compile(optimizer='sgd', loss='mse')
        self.compiled = True

    def load_weights(self, filepath):
        self.model.load_weights(filepath)

    def save_weights(self, filepath, overwrite=False):
        self.model.save_weights(filepath, overwrite=overwrite)

    def get_weights_flat(self,weights):
        weights_flat = np.zeros(self.num_weights)

        pos = 0
        for i_layer, size in enumerate(self.sizes):
            weights_flat[pos:pos+size] = weights[i_layer].flatten()
            pos += size
        return weights_flat
        
    def get_weights_list(self,weights_flat):
        weights = []
        pos = 0
        for i_layer, size in enumerate(self.sizes):
            arr = weights_flat[pos:pos+size].reshape(self.shapes[i_layer])
            weights.append(arr)
            pos += size
        return weights          

    def reset_states(self):
        self.recent_observation = None
        self.recent_action = None

    def select_action(self, state, stochastic=False):
        batch = np.array([state])
        if self.processor is not None:
            batch = self.processor.process_state_batch(batch)

        action = self.model.predict_on_batch(batch).flatten()
        if stochastic or self.training:
            return np.random.choice(np.arange(self.nb_actions), p=np.exp(action) / np.sum(np.exp(action)))
        return np.argmax(action)
    
    def update_theta(self,theta):
        if (theta is not None):
            assert theta.shape == self.theta.shape, "Invalid theta, shape is {0} but should be {1}".format(theta.shape,self.theta.shape)
            assert (not np.isnan(theta).any()), "Invalid theta, NaN encountered"
            assert (theta[self.num_weights:] >= 0.).all(), "Invalid theta, standard deviations must be nonnegative"            
            self.theta = theta
        else:
            means = np.ones(self.num_weights) * self.init_mean
            stdevs = np.ones(self.num_weights) * self.init_stdev
            self.theta = np.hstack((means,stdevs))

    def choose_weights(self):
        mean = self.theta[:self.num_weights]
        std = self.theta[self.num_weights:]
        weights_flat = std * np.random.randn(self.num_weights) + mean

        sampled_weights = self.get_weights_list(weights_flat)
        self.model.set_weights(sampled_weights)

    def forward(self, observation):
        # Select an action.
        state = self.memory.get_recent_state(observation)
        action = self.select_action(state)
        if self.processor is not None:
            action = self.processor.process_action(action)

        # Book-keeping.
        self.recent_observation = observation
        self.recent_action = action

        return action
         
    def backward(self, reward, terminal):
        # Store most recent experience in memory.
        if self.step % self.memory_interval == 0:
            self.memory.append(self.recent_observation, self.recent_action, reward, terminal,
                               training=self.training)

        metrics = [np.nan for _ in self.metrics_names]
        if not self.training:
            # We're done here. No need to update the experience memory since we only use the working
            # memory to obtain the state over the most recent observations.
            return metrics

        if terminal:
            params = self.get_weights_flat(self.model.get_weights())
            self.memory.finalize_episode(params)

            if self.step > self.nb_steps_warmup and self.episode % self.train_interval == 0:
                params, reward_totals = self.memory.sample(self.batch_size)
                best_idx = np.argsort(np.array(reward_totals))[-self.num_best:]
                best = np.vstack([params[i] for i in best_idx])

                if reward_totals[best_idx[-1]] > self.best_seen[0]:
                    self.best_seen = (reward_totals[best_idx[-1]], params[best_idx[-1]])
                    
                metrics = [np.mean(np.array(reward_totals)[best_idx])]
                if self.processor is not None:
                    metrics += self.processor.metrics
                min_std = self.noise_ampl * np.exp(-self.step * self.noise_decay_const)
                
                mean = np.mean(best, axis=0)
                std = np.std(best, axis=0) + min_std
                new_theta = np.hstack((mean, std))
                self.update_theta(new_theta)
            self.choose_weights()
            self.episode += 1
        return metrics

    def _on_train_end(self):
        self.model.set_weights(self.get_weights_list(self.best_seen[1]))

    @property
    def metrics_names(self):
        names = ['mean_best_reward']
        if self.processor is not None:
            names += self.processor.metrics_names[:]
        return names
