from __future__ import division
import pytest
import numpy as np
from numpy.testing import assert_allclose
import random

from keras.models import Model, Sequential
from keras.layers import Input, merge, Dense, Flatten

from rl.agents.dqn import NAFLayer, DQNAgent
from rl.memory import SequentialMemory
from rl.core import MultiInputProcessor


class Env(object):
    def __init__(self, observation_shape):
        self.observation_shape = observation_shape

    def step(self, action):
        return self._get_obs(), random.choice([0, 1]), False, {}

    def reset(self):
        return self._get_obs()

    def _get_obs(self):
        if type(self.observation_shape) is list:
            return [np.random.random(s) for s in self.observation_shape]
        else:
            return np.random.random(self.observation_shape)


def test_single_dqn_input():
    model = Sequential()
    model.add(Flatten(input_shape=(1, 3)))
    model.add(Dense(2))

    memory = SequentialMemory(limit=10, window_length=1)
    for double_dqn in (True, False):
        agent = DQNAgent(model, memory=memory, nb_actions=2, nb_steps_warmup=5, batch_size=4,
                         enable_double_dqn=double_dqn)
        agent.compile('sgd')
        agent.fit(Env((3,)), nb_steps=10)


def test_multi_dqn_input():
    input1 = Input(shape=(2, 3))
    input2 = Input(shape=(2, 4))
    x = merge([input1, input2], mode='concat')
    x = Flatten()(x)
    x = Dense(2)(x)
    model = Model(input=[input1, input2], output=x)

    memory = SequentialMemory(limit=10, window_length=2)
    processor = MultiInputProcessor(nb_inputs=2)
    for double_dqn in (True, False):
        agent = DQNAgent(model, memory=memory, nb_actions=2, nb_steps_warmup=5, batch_size=5,
                         processor=processor, enable_double_dqn=double_dqn)
        agent.compile('sgd')
        agent.fit(Env([(3,), (4,)]), nb_steps=10)


def test_naf_layer_full():
    batch_size = 2
    for nb_actions in (1, 3):
        # Construct single model with NAF as the only layer, hence it is fully deterministic
        # since no weights are used, which would be randomly initialized.
        L_flat_input = Input(shape=((nb_actions * nb_actions + nb_actions) // 2,))
        mu_input = Input(shape=(nb_actions,))
        action_input = Input(shape=(nb_actions,))
        x = merge([L_flat_input, mu_input, action_input], mode='concat')
        x = NAFLayer(nb_actions, mode='full')(x)
        model = Model(input=[L_flat_input, mu_input, action_input], output=x)
        model.compile(loss='mse', optimizer='sgd')
        
        # Create random test data.
        L_flat = np.random.random((batch_size, (nb_actions * nb_actions + nb_actions) // 2)).astype('float32')
        mu = np.random.random((batch_size, nb_actions)).astype('float32')
        action = np.random.random((batch_size, nb_actions)).astype('float32')

        # Perform reference computations in numpy since these are much easier to verify.
        L = np.zeros((batch_size, nb_actions, nb_actions)).astype('float32')
        LT = np.copy(L)
        for l, l_T, l_flat in zip(L, LT, L_flat):
            l[np.tril_indices(nb_actions)] = l_flat
            l[np.diag_indices(nb_actions)] = np.exp(l[np.diag_indices(nb_actions)])
            l_T[:, :] = l.T
        P = np.array([np.dot(l, l_T) for l, l_T in zip(L, LT)]).astype('float32')
        A_ref = np.array([np.dot(np.dot(a - m, p), a - m) for a, m, p in zip(action, mu, P)]).astype('float32')
        A_ref *= -.5

        # Finally, compute the output of the net, which should be identical to the previously
        # computed reference.
        A_net = model.predict([L_flat, mu, action]).flatten()
        assert_allclose(A_net, A_ref, rtol=1e-5)


def test_naf_layer_diag():
    batch_size = 2
    for nb_actions in (1, 3):
        # Construct single model with NAF as the only layer, hence it is fully deterministic
        # since no weights are used, which would be randomly initialized.
        L_flat_input = Input(shape=(nb_actions,))
        mu_input = Input(shape=(nb_actions,))
        action_input = Input(shape=(nb_actions,))
        x = merge([L_flat_input, mu_input, action_input], mode='concat')
        x = NAFLayer(nb_actions, mode='diag')(x)
        model = Model(input=[L_flat_input, mu_input, action_input], output=x)
        model.compile(loss='mse', optimizer='sgd')
        
        # Create random test data.
        L_flat = np.random.random((batch_size, nb_actions)).astype('float32')
        mu = np.random.random((batch_size, nb_actions)).astype('float32')
        action = np.random.random((batch_size, nb_actions)).astype('float32')

        # Perform reference computations in numpy since these are much easier to verify.
        P = np.zeros((batch_size, nb_actions, nb_actions)).astype('float32')
        for p, l_flat in zip(P, L_flat):
            p[np.diag_indices(nb_actions)] = l_flat
        print(P, L_flat)
        A_ref = np.array([np.dot(np.dot(a - m, p), a - m) for a, m, p in zip(action, mu, P)]).astype('float32')
        A_ref *= -.5

        # Finally, compute the output of the net, which should be identical to the previously
        # computed reference.
        A_net = model.predict([L_flat, mu, action]).flatten()
        assert_allclose(A_net, A_ref, rtol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__])
