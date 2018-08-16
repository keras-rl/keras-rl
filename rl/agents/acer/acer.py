from __future__ import division
import warnings
from collections import namedtuple
import os

import numpy as np
import keras.backend as K
import keras.optimizers as optimizers

from rl.core import Agent
from rl.util import *
from time import time, sleep

# TODO : Add different warnings and exceptions
# TODO : change episodic memory implementation
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'done', 'mus'))

def returns(R):
    assert (len(K.int_shape(R))==1)
    return K.mean(R, axis=0, keepdims=False)

def convert_q_retrace_to_batch(val, nenvs, nsteps):
    qret = []
    for j in range(nenvs):
        for i in range(nsteps):
            qret.append(val[j][i])
    return qret

def q_retrace(R, D, q_i, v, rho_i, gamma, nenvs, nsteps):
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

    _R = K.reshape(R, shape=(nenvs, nsteps))
    _D = K.reshape(D, shape=(nenvs, nsteps))
    _q_i = K.reshape(q_i, shape=(nenvs, nsteps))
    _v = K.reshape(v, shape=(nenvs, nsteps))
    _rho_bar = K.reshape(rho_bar, shape=(nenvs, nsteps))

    # qrets = np.zeros((nenvs, nsteps))
    qrets = [[] for _ in range(nenvs)]

    for j in range(nenvs):
        qret = _v[j,-1] * (1 - _D[j,-1])
        q = []
        for i in reversed(range(nsteps)):
            qret = _R[j][i] + gamma * qret * (1 - _D[j][i])
            q.append(qret)
            qret = (_rho_bar[j][i] * (qret - _q_i[j][i])) + _v[j][i]
        qrets[j] = q[::-1]
    # print(len(qrets), len(qrets[0]))
    return convert_q_retrace_to_batch(qrets, nenvs, nsteps)

def get_by_index(x, idx, shape=None):
    if shape is None:
        shape = K.int_shape(x)
    idx_flattened = K.arange(0, shape[0]) * shape[1] + idx
    y = K.gather(K.reshape(x, [-1]),  # flatten input
                 idx_flattened)  # use flattened indices
    return y

class ACERAgent(Agent):
    def __init__(self, memory, model_fn, nb_actions, obs_shape, policy, gamma=0.99, nenvs=1, memory_interval=1,
                 batch_size = 4, on_policy=False, replay_ratio = 4, replay_start=2000, max_gradient_norm = 40,
                 nb_warmup_steps = 40, trace_decay=1, trace_max=10, trust_region=True, trust_region_decay=0.99, 
                 trust_region_thresold = 1., nsteps = 20, eps= 1e-5, entropy_weight = 1e-3, value_weight = 0.5,
                 **kwargs):
        super(ACERAgent, self).__init__(**kwargs)         

        # Parameters        
        self.nb_actions = nb_actions
        self.obs_shape = obs_shape
        self.policy = policy
        self.gamma = gamma
        self.nenvs = nenvs
        self.nsteps = nsteps
        self.memory_interval = memory_interval
        self.batch_size = batch_size
        self.on_policy = on_policy
        self.replay_ratio = replay_ratio
        self.replay_start = replay_start
        self.nb_warmup_steps = nb_warmup_steps

        # ACER specific parameters
        self.max_gradient_norm = max_gradient_norm
        self.trust_region = trust_region
        self.trace_max = trace_max
        self.trace_decay = trace_decay
        self.trust_region_decay = trust_region_decay
        self.trust_region_thresold = trust_region_thresold
        self.entropy_weight = entropy_weight
        self.value_weight = value_weight
        self.eps = eps

        # Model
        self.memory = memory
        self.model_fn = model_fn
        self.model, _, _ = self.make_model(self.nsteps, model_fn)
        self.average_model, _, _ = self.make_model(self.nsteps, model_fn)
        self.step_model, self.step_model_input, self.step_model_output = self.make_model(1, model_fn, 'step_input')
        self.test_model = None
        self.step_fn = self.make_function(self.step_model_input, self.step_model_output)
        self.update_step_model_weights()
        self.update_average_model_weights()
        self.testing = False

        self.nbatch = self.nenvs * self.nsteps
        self.trajectory = [[] for _ in range(self.nenvs)]
        
        # State
        self.t_start = 0
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

        # print ('Compiling the Agent')
        # print (K.backend())
        inp = self.model.input
        Q, mus = self.model(inp)
        _, avg_mus = self.average_model(inp)

        old_mus = K.placeholder(shape=(self.nbatch, self.nb_actions))
        A = K.placeholder(shape=[self.nbatch], dtype='int32')
        R = K.placeholder(shape=[self.nbatch])
        D = K.placeholder(shape=[self.nbatch])
        rho = mus/(old_mus + self.eps)

        V = K.mean(mus*Q, axis=-1, keepdims=False)

        Q_i = get_by_index(Q, A)

        # Note shape is sent to deal with the no shape error in keras (theano)
        rho_i = get_by_index(rho, A, shape=(self.nbatch, self.nb_actions))

        Qret = q_retrace(R, D, Q_i, V, rho_i, self.gamma, self.nenvs, self.nsteps)
        assert len(Qret) == self.nbatch

        f_i = get_by_index(mus, A)
        # print ('Qret and f_i calculated')

        # Policy gradient loss
        adv = Qret - V
        logf = K.log(f_i + self.eps)
        gain_f = logf * K.stop_gradient(adv * K.clip(rho_i, min_value=0, max_value=self.trace_max))
        loss_f = -K.sum(gain_f)
        # print ('Loss f calculated')

        # Bias correction for the truncation
        adv_bc = Q - K.reshape(V, (self.nbatch, 1))
        logf_bc = K.log(mus + self.eps)
        gain_bc = K.sum(logf_bc * K.stop_gradient(adv_bc * K.relu(1. - self.trace_max / (rho + self.eps)) * mus), axis=-1)
        loss_bc = -K.sum(gain_bc)

        loss_policy = loss_bc + loss_f
        # print ('policy loss calculated')

        # Add Entropy
        entropy = -K.mean(K.sum(K.log(mus)*mus, axis=1))

        # Define KL divergence
        kl = K.sum(-avg_mus * (K.log(mus + self.eps) - K.log(avg_mus+ self.eps)), axis=1)
        
        # Add trust region
        if self.trust_region:
            g = K.gradients(loss_policy * self.nsteps * self.nenvs, mus)
            k = - avg_mus / (mus + self.eps)
            k_dot_g = K.sum(k*g, axis=-1)
            k_dot_k = K.sum(k*k, axis=-1)
            coeffs = K.relu((k_dot_g - self.trust_region_thresold)/(k_dot_k + self.eps))
            trust_err = K.mean(K.stop_gradient(coeffs) * kl)
            loss_policy += trust_err

        loss_policy -= self.entropy_weight*entropy
        # print len(Qret)
        # print Qret
        # print (K.int_shape(Q_i), len(Qret))
        loss_value = 0.5 * K.mean(K.square(K.stop_gradient(Qret) - Q_i))
        
        total_loss = loss_policy + self.value_weight*loss_value

        inputs = [inp]
        inputs = inputs + [old_mus, A, R, D]
        # metrics = K.mean(f_i * R)

        updates = self.optimizer.get_updates(total_loss, self.model.trainable_weights)

        # Finally, combine it all into a callable function.
        if K.backend() == 'tensorflow':
            self.train_fn = K.function(inputs + [K.learning_phase()],
                                       [total_loss], updates=updates)
        else:
            if self.uses_learning_phase:
                inputs += [K.learning_phase()]
            self.train_fn = K.function(inputs, [total_loss], updates=updates)
        # print ('Agent Compiled')
        self.compiled = True

    def make_model(self, nsteps, model_fn, name=None, nenvs=None):
        if nenvs == None:
            nenvs = self.nenvs
        shape = (nenvs * nsteps,) + self.obs_shape
        inp = K.placeholder(shape=shape)
        if name == None:
            return model_fn(inp)
        else:
            return model_fn(inp, name)

    def make_function(self, inp, out):
        return K.function(inputs=inp, outputs=out)

    def update_step_model_weights(self):
        self.step_model.set_weights(self.model.get_weights())

    def update_test_model_weights(self):
        if not self.test_model:
            raise ValueError('Test model not defined')
        self.test_model.set_weights(self.model.get_weights())

    def update_average_model_weights(self):
        model_weights = self.model.get_weights()
        average_model_weights = self.average_model.get_weights()
        assert len(model_weights) == len(average_model_weights)
        weights = []
        for i in range(len(model_weights)):
            w = model_weights[i] * (1 -self.trust_region_decay) +\
                average_model_weights[i] * self.trust_region_decay
            weights.append(w)
        self.average_model.set_weights(weights)

    def forward(self, observation):
        # Select an action.
        if (len(self.obs_shape) + 1) == len(observation.shape):
            state = np.asarray(observation, dtype=np.float32)
            if self.testing is True:
                self.testing = False
        elif (len(self.obs_shape)) == len(observation.shape):
            # This is for testing the game
            self.testing = True
            if not self.test_model:
                self.test_model, inp, out = self.make_model(nsteps=1, model_fn=self.model_fn, nenvs=1, name='test_input')
                self.test_fn = self.make_function(inp, out)
                self.testing = 'True'
                self.update_test_model_weights()

            state = np.asarray([observation], dtype=np.float32)
        else:
            raise ValueError('The dimention of state is inconsistent with the input dimention')
        if self.testing:
            _, mus = self.test_fn([state])
            self.recent_observation = observation
            self.recent_mus = mus[0]
            self.recent_action = self.policy.select_action(self.nb_actions, self.recent_mus, testing=True)
            return self.recent_action
        else:
            _, mus = self.step_fn([state])
            self.recent_observation = observation
            self.recent_mus = mus
            self.recent_action = []
            for i in range(len(self.recent_mus)):
                action = self.policy.select_action(self.nb_actions, self.recent_mus[i])
                self.recent_action.append(action)
            return self.recent_action

    # We do not need this function. However, for generality we will define it.
    def update_target_models_hard(self):
        pass

    @property
    def metrics_names(self):
        names = self.model.metrics_names[:]
        return names

    def load_weights(self, filepath):
        filename, extension = os.path.splitext(filepath)
        model_filepath = filename + '_acer_model' + extension
        self.model.load_weights(model_filepath)
        self.update_step_model_weights()
        self.update_average_model_weights()
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
            self.average_model.reset_states()
            self.step_model.reset_states()
            if self.testing:
                self.test_model.reset_states()

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
            self.trajectory = [[] for _ in range(self.nenvs)]
            return metrics

        if self.step % self.memory_interval == 0:
            for i in range(self.nenvs):
                self.trajectory[i].append(Transition(self.recent_observation[i], self.recent_action[i],
                                       reward[i], terminal[i], self.recent_mus[i]))
            self.t_start = self.t_start + 1
        
        # Use this for learning from experience replay
        can_learn_from_memory = self.step > self.replay_start and (not self.on_policy)

        if self.t_start % self.nsteps == 0:
            self.memory.put(self.trajectory)
            # Learn on_policy
            self.learn_from_trajectory()
            if can_learn_from_memory:
                # Learn off policy
                self.learn_from_memory()
                # sleep(10)

        return metrics

    def learn_from_trajectory(self, trajectory=None): 
        # If trajectory is None, we will learn from recent experience
        # Else we will learn from the memory batch       
        if trajectory is None:
            trajectory = self.trajectory
            self.trajectory = [[] for _ in range(self.nenvs)]
        
        assert len(trajectory) == self.nenvs
        obs, old_mus, A, R, D = [], [], [], [], []
        for traj in trajectory:
            obs_t, old_mus_t, A_t, R_t, D_t = [], [], [], [], []
            for i in range(len(traj)):
                action = traj[i].action
                obs_t.append(traj[i].state)
                old_mus_t.append(traj[i].mus)
                A_t.append(action)
                R_t.append(traj[i].reward)
                D_t.append(traj[i].done)
            obs.append(obs_t)
            old_mus.append(old_mus_t)
            A.append(A_t)
            R.append(R_t)
            D.append(D_t)
        
        # Convert the list to numpy array
        obs = np.asarray(obs)
        old_mus = np.asarray(old_mus)
        A = np.asarray(A, dtype=np.uint8)
        R = np.asarray(R)
        D = np.asarray(D)

        # Reshape for suitable inputs
        obs = np.reshape(obs, newshape=(self.nbatch,) + self.obs_shape)
        old_mus = np.reshape(old_mus, newshape=(self.nbatch, self.nb_actions))
        A = np.reshape(A, newshape=[self.nbatch])
        R = np.reshape(R, newshape=[self.nbatch])
        D = np.reshape(D, newshape=[self.nbatch])
        # print obs.shape, old_mus.shape, A.shape, R.shape, D.shape
        # Training in the model
        self.train_fn([obs, old_mus, A, R, D])
        self.update_step_model_weights()
        self.update_average_model_weights()

    def learn_from_memory(self):
        n = np.random.poisson(self.replay_ratio)
        for _ in range(n):
            batch = self.memory.get(self.batch_size)
            for trajectory in batch:
                self.learn_from_trajectory(trajectory)
