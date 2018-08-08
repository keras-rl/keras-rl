from __future__ import division
import warnings
from collections import namedtuple
# import multiprocessing as mp

import numpy as np
import keras.backend as K
import keras.optimizers as optimizers

from rl.core import Agent
from rl.util import *
from time import time

# TODO : Add different warnings and exceptions

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'done', 'mus'))

def returns(R):
    assert (len(K.int_shape(R))==1)
    return K.mean(R, axis=0, keepdims=False)

def q_retrace(R, D, q_i, v, rho_i, gamma):
    """
    Calculates q_retrace targets
    :param R: Rewards
    :param D: Dones
    :param q_i: Q values for actions taken
    :param v: V values
    :param rho_i: Importance weight for each action
    :return: Q_retrace values
    """
    assert (len(K.int_shape(R)) == 1)
    assert (len(K.int_shape(D)) == 1)
    rho_bar = K.clip(rho_i,min_value=0, max_value=1.0)
    qrets = []
    qret = v[-1] * (1 - D[-1])
    l = K.int_shape(R)[0]
    for i in reversed(range(l)):
        qret = R[i] + gamma * qret * (1 - D[i])
        qrets.append(qret)
        qret = (rho_bar[i] * (qret - q_i[i])) + v[i]
    return qrets

def get_by_index(x, idx, shape=None):
    if shape is None:
        shape = K.int_shape(x)
    idx_flattened = K.arange(0, shape[0]) * shape[1] + idx
    y = K.gather(K.reshape(x, [-1]),  # flatten input
                 idx_flattened)  # use flattened indices
    return y

class ACERAgent(Agent):
    def __init__(self, model_fn, nb_actions, obs_shape, policy, gamma=0.99, nenvs=1, memory_interval=1,
                 on_policy=False, replay_ratio = 4, num_process=1, replay_start=200, nb_warmup_steps = 40,
                 trace_decay=1, trace_max=10, trust_region=True, trust_region_decay=0.99, 
                 trust_region_thresold = 1., nsteps = 20, eps= 1e-5,
                 entropy_weight = 1e-4, value_weight = 0.5,max_gradient_norm = 40, **kwargs):
        super(ACERAgent, self).__init__(**kwargs)         

        # Parameters
        self.nb_actions = nb_actions
        self.obs_shape = obs_shape
        self.gamma = gamma
        self.policy = policy
        self.on_policy = on_policy
        self.replay_ratio = replay_ratio
        self.replay_start = replay_start
        self.num_process = num_process
        self.trace_max = trace_max
        self.trace_decay = trace_decay
        self.trust_region_decay = trust_region_decay
        self.trust_region_thresold = trust_region_thresold
        self.entropy_weight = entropy_weight
        self.value_weight = value_weight
        self.max_gradient_norm = max_gradient_norm
        self.nsteps = nsteps
        self.nenvs = nenvs
        self.eps = eps
        self.memory_interval = memory_interval
        self.nb_warmup_steps = 40
        # Model        

        self.model, _ , _ = self.make_model(self.nsteps, model_fn)
        self.step_model, self.step_model_input, self.step_model_output = self.make_model(1, model_fn, 'step_input')
        self.step_fn = self.make_step_function()
        self.update_step_model_weights()

        self.nbatch = self.nenvs * self.nsteps
        self.trajectory = []
        
        self.t_start = 0

        self.terminal = False

        self.compiled = False
        self.reset_states()

@property
    def uses_learning_phase(self):
        return self.model.uses_learning_phase

    def compile(self, optimizer, metrics=[]):

        if type(optimizer) in (list, tuple):
            if len(optimizer) >= 2:
                raise ValueError('More than one optimizers provided. Please only provide a single optimizer.')            
        else:
            self.optimizer = optimizer
        if type(optimizer) is str:
            self.optimizer = optimizers.get(optimizer)

        if len(metrics) != 0:
            raise ValueError('Please add your metrics to the computation graph. Current implementation supports None')

        self.model.metrics_names = ['Avg_Rewards']

        print ('Compiling the Agent')
        print (K.backend())
        inp = self.model.input
        Q, mus = self.model(inp)

        old_mus = K.placeholder(shape=(self.nbatch, self.nb_actions))
        A = K.placeholder(shape=[self.nbatch], dtype='int32')
        R = K.placeholder(shape=[self.nbatch])
        D = K.placeholder(shape=[self.nbatch])
        rho = mus/old_mus

        V = K.mean(mus*Q, axis=1, keepdims=False)

        Q_i = get_by_index(Q, A)
        rho_i = get_by_index(rho, A, shape=(self.nbatch, self.nb_actions))

        Qret = q_retrace(R, D, Q_i, V, rho_i, self.gamma)
        f_i = get_by_index(mus, A)
        # print ('Qret and f_i calculated')

        # Policy gradient loss
        adv = Qret - V
        logf = K.log(f_i + self.eps)
        gain_f = logf * K.stop_gradient(adv * K.clip(rho_i, min_value=0, max_value=self.trace_max))
        loss_f = -K.mean(gain_f)
        # print ('Loss f calculated')

        # Bias correction for the truncation
        adv_bc = Q - K.reshape(V, (self.nbatch, 1))
        logf_bc = K.log(mus + self.eps)
        gain_bc = K.sum(logf_bc * K.stop_gradient(adv_bc * K.relu(1. - self.trace_max / (rho + self.eps)) * mus), axis=1)
        loss_bc = -K.mean(gain_bc)

        loss_policy = loss_bc + loss_f
        # print ('policy loss calculated')

        # Add Entropy
        entropy = -K.mean(K.sum(K.log(mus)*mus, axis=1))

        loss_policy -= self.entropy_weight*entropy
        loss_value = 0.5 * K.mean(K.square(K.stop_gradient(Qret) - Q_i))

        total_loss = loss_policy + self.value_weight*loss_value

        inputs = [inp]
        inputs = inputs + [old_mus, A, R, D]
        # metrics = returns(R)
        metrics = K.mean(f_i * R)

        # Add trust region

        updates = self.optimizer.get_updates(total_loss, self.model.trainable_weights)

        # Finally, combine it all into a callable function.
        if K.backend() == 'tensorflow':
            self.train_fn = K.function(inputs + [K.learning_phase()],
                                       [metrics], updates=updates)
        else:
            if self.uses_learning_phase:
                inputs += [K.learning_phase()]
            self.train_fn = K.function(inputs, [metrics], updates=updates)
        print ('Agent Compiled')
        self.compiled = True

    def make_model(self, nsteps, model_fn, name=None):
        shape = (self.nenvs * nsteps,) + self.obs_shape
        inp = K.placeholder(shape=shape)
        if name == None:
            return model_fn(inp)
        else:
            return model_fn(inp, name)

    def make_step_function(self):
        # inp = self.step_model.input
        # out = self.step_model(inp)
        return K.function(inputs=self.step_model_input, outputs=self.step_model_output)

    def update_step_model_weights(self):
        self.step_model.set_weights(self.model.get_weights())

    
    def forward(self, observation):
        # Select an action.
        state = np.asarray([observation], dtype=np.float32)
        _, mus = self.step_fn([state])
        self.recent_observation = observation
        # To deal with error : ValueError: probabilities do not sum to 1
        self.recent_mus = mus[0]/sum(mus[0])
        action = self.policy.select_action(self.nb_actions, self.recent_mus)
        self.recent_action = action
        return action

    @property
    def metrics_names(self):
        names = self.model.metrics_names[:]
        return names

    def load_weights(self, filepath):
        filename, extension = os.path.splitext(filepath)
        model_filepath = filename + '_acer_model' + extension
        self.model.load_weights(model_filepath)
        self.update_target_models_hard()

    def save_weights(self, filepath, overwrite=False):
        filename, extension = os.path.splitext(filepath)
        model_filepath = filename + '_acer_model' + extension
        self.model.save_weights(model_filepath, overwrite=overwrite)

    # TODO : Check reset states
    def reset_states(self):
        self.recent_action = None
        self.recent_observation = None
        if self.compiled:
            self.model.reset_states()

    @property
    def layers(self):
        return self.model.layers[:]

    @property
    def metrics_names(self):
        names = self.model.metrics_names[:]
        if self.processor is not None:
            names += self.processor.metrics_names[:]
        return names

    def backward(self, reward, terminal=False):


        metrics = [np.nan for _ in self.metrics_names]

        if not self.training:
            # We're done here. No need to update the experience memory since we only use the working
            # memory to obtain the state over the most recent observations.
            self.trajectory = []
            return metrics

        if self.step % self.memory_interval == 0 or self.terminal:
            if self.terminal:
                self.terminal = False
                return metrics
            else:
                self.trajectory.append(Transition(self.recent_observation, self.recent_action,
                                       reward, terminal, self.recent_mus))
                self.t_start = self.t_start + 1

        if terminal:
            self.terminal = True
        
        # Use this for learning from experience replay
        can_train_either = self.step > self.nb_warmup_steps

        if self.t_start % self.nsteps == 0:
            assert len(self.trajectory) == self.nsteps
            obs, old_mus, A, R, D = [], [], [], [], []
            for i in range(len(self.trajectory)):
                action = self.trajectory[i].action
                obs.append(self.trajectory[i].state)
                # TODO : Add off policy
                old_mus.append(self.trajectory[i].mus)
                A.append(action)
                R.append(self.trajectory[i].reward)
                D.append(self.trajectory[i].done)
            self.train_fn([obs, old_mus, A, R, D])
            self.update_step_model_weights()
            self.trajectory = []
        return metrics
