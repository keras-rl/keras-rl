from __future__ import division
import warnings
from collections import namedtuple
# import multiprocessing as mp

import keras.backend as K
import keras.optimizers as optimizers
from keras.models import Model
from keras.layers import Lambda, Input, Layer, Dense

from rl.core import Agent
from rl.util import *

# TODO : Add different warnings and exceptions

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'done', 'mus'))

def mean_q(y_true, y_pred):
    return K.mean(K.max(y_pred, axis=-1))

class ACERAgent(Agent):
    def __init__(self, model, nb_actions, gamma=0.99, batch_size=32,
                 on_policy=False, replay_ratio = 4, num_process=1, replay_start=2000, 
                 trace_decay=1, trace_max=10, trust_region=True, trust_region_decay=0.99, 
                 trust_region_thresold = 1., lr=1e-5, rmsprop_decay = 0.99, nsteps = 20,
                 entropy_weight = 1e-4, value_weight = 0.5,max_gradient_norm = 40, **kwargs):
        super(ACERAgent, self).__init__(**kwargs)         

        # Parameters
        self.nb_actions = nb_actions
        # self.memory = memory
        self.gamma = gamma
        self.batch_size = 32
        self.on_policy = on_policy
        self.replay_ratio = replay_ratio
        # self.replay_start = replay_start
        self.num_process = num_process
        # self.trace_max = trace_max
        # self.trace_decay = trace_decay
        # self.trust_region_decay = trust_region_decay
        # self.trust_region_thresold = trust_region_thresold
        self.lr = lr
        # self.rmsprop_decay = rmsprop_decay
        self.entropy_weight = entropy_weight
        self.value_weight = value_weight
        self.max_gradient_norm = max_gradient_norm
        self.nsteps = nsteps
        self.nenv = 1
        # Model        
        self.model = model

        self.nbatch = nenv * nsteps
        self.trajectory = []
        
        self.t_start = 0

        self.reset_states()

    @property
    def uses_learning_phase(self):
        return self.actor.uses_learning_phase or self.critic.uses_learning_phase

    def compile(self, optimizer, metrics=[]):

        if type(optimizer) in (list, tuple):
            if len(optimizer) >= 2:
                raise ValueError('More than one optimizers provided. Please only provide a single optimizer.')            
        else:
            self.optimizer = optimizer
        if type(optimizer) is str:
            self.optimizer = optimizers.get(optimizer)

        # TODO : Check metrics part
        # if len(metrics) == 2 and hasattr(metrics[0], '__len__') and hasattr(metrics[1], '__len__'):
        #     actor_metrics, critic_metrics = metrics
        # else:
        #     actor_metrics = critic_metrics = metrics

        mus = K.placeholder(shape=(self.nbatch, self.nb_actions))
        old_mus = K.placeholder(shape=(self.nbatch, self.nb_actions))
        Q = K.placeholder(shape=(self.nbatch, self.nb_actions))
        A = K.placeholder(shape=[self.nbatch])
        R = K.placeholder(shape=[self.nbatch])
        D = K.placeholder(shape=[self.nbatch])

        V = K.mean(Q, axis=1)
        f_i = get_by_index(mus, A)
        Q_i = get_by_index(Qs, A)

        # if on_policy rho is 1
        rho = K.stop_gradient(mus/old_mus)
        rho_i = get_by_index(rho, A)

        Qret = self.q_retrace(R, D, Q_i, V)

        # Policy gradient loss
        adv = Qret - Vs
        logf_i = K.log(f_i + self.eps)
        gain_f = logf * K.stop_gradient(adv * K.clip(rho_i, max_value=self.c))
        loss_f = -K.mean(gain_f)

        # Bias correction for the truncation
        adv_bc = Q - K.reshape(V, (self.nbatch, 1))
        logf_bc = K.log(mus + self.eps)
        gain_bc = K.sum(logf_bc * K.stop_gradient(adv_bc * K.relu(1. - self.c / (rho + self.eps)) * mus), axis=1)
        loss_bc = -K.mean(gain_bc)

        loss_policy = loss_bc + loss_f
        
        # Add Entropy
        entropy = -K.mean(K.sum(K.log(mus)*mus, axis=1))

        loss_policy -= self.entropy_weight*entropy
        loss_value = 0.5 * K.mean(K.square(K.stop_gradient(Qret) - Q_i))

        total_loss = loss_policy + self.value_weight*loss_value

        inputs = []
        for i in self.model.input:
            inputs.append(i)
        inputs = inputs + [mus, old_mus,  Q, A, R, D]

        # Add trust region

        updates = self.optimizer.get_updates(total_loss, self.model.trainable_weights)

        # Finally, combine it all into a callable function.
        if K.backend() == 'tensorflow':
            self.train_fn = K.function(inputs + [K.learning_phase()],
                                             [self.actor(inputs)], updates=updates)
        else:
            if self.uses_learning_phase:
                inputs += [K.learning_phase()]
            self.train_fn = K.function(inputs, [self.actor(inputs)], updates=updates)
        self.compiled = True


    def forward(self, observation):
        # Select an action.
        state = self.memory.get_recent_state(observation)
        mus = self.actor(state)  

        self.recent_observation = observation
        self.recent_mus = mus
        action = self.policy.select_action(actions)

        return action

    @property
    def metrics_names(self):
        names = self.critic.metrics_names[:]
        return names

    def load_weights(self, filepath):
        filename, extension = os.path.splitext(filepath)
        actor_filepath = filename + '_actor' + extension
        critic_filepath = filename + '_critic' + extension
        self.actor.load_weights(actor_filepath)
        self.critic.load_weights(critic_filepath)
        self.update_target_models_hard()

    def save_weights(self, filepath, overwrite=False):
        filename, extension = os.path.splitext(filepath)
        actor_filepath = filename + '_actor' + extension
        critic_filepath = filename + '_critic' + extension
        self.actor.save_weights(actor_filepath, overwrite=overwrite)
        self.critic.save_weights(critic_filepath, overwrite=overwrite)

    # TODO : Check reset states
    def reset_states(self):
        self.recent_action = None
        self.recent_observation = None
        if self.compiled:
            self.model.reset_states()

    @property
    def layers(self):
        return self.actor.layers[:] + self.critic.layers[:]

    @property
    def metrics_names(self):
        names = self.critic.metrics_names[:]
        if self.processor is not None:
            names += self.processor.metrics_names[:]
        return names

    def backward(self, reward, terminal=False):

        if self.step % self.memory_interval == 0 or self.terminal:
            if self.terminal:
                self.terminal = False
            else:
                self.trajectory.append(Transition(self.recent_observation, self.recent_action,
                                       reward, terminal, self.recent_mus))
                self.t_start = self.t_start + 1

        metrics = [np.nan for _ in self.metrics_names]

        if not self.training:
            # We're done here. No need to update the experience memory since we only use the working
            # memory to obtain the state over the most recent observations.
            self.trajectory = []
            return metrics

        if terminal:
            self.terminal = True
        
        # Use this for learning from experience replay
        can_train_either = self.step > self.nb_warmup_steps

        if self.t_start % self.nsteps == 0:
            assert len(self.trajectory) == self.nsteps
            obs, mus, old_mus, Q, A, R, D = [], [], [], [], [], [], []
            for i in range(len(self.trajectory)):
                obs.append(trajectory[i].state)
                mus.append(trajectory[i].mus)
                old_mus.append(trajectory[i].mus)
                Q.append(self.model(K.variable([trajectory[i]]))[0,0])
                A.append(trajectory[i].action)
                R.append(trajectory[i].reward)
                D.append(trajectory[i].done)
            self.train_fn([obs, mus, old_mus,  Q, A, R, D])
            print("Trajectory " , self.t_start / self.nsteps, " optimised")
