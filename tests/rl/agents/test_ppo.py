from __future__ import division
from __future__ import absolute_import

import pytest
import numpy as np
import gym

import keras.backend as K

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, merge

from rl.agents.ppo import PPOAgent
from rl.memory import SequentialMemory

from rl.random import IndependentGaussianProcess

# from https://github.com/MorvanZhou/train-robot-arm-from-scratch
# redacted until libre license confirmed (or alternative found)

def test_ppo_basic():
    env = ArmEnv()
    critic = Sequential()
    critic.add(Flatten(input_shape=(3, 9)))
    critic.add(Dense(32))
    critic.add(Activation('relu'))
    critic.add(Dense(32))
    critic.add(Activation('relu'))
    critic.add(Dense(32))
    critic.add(Activation('relu'))
    critic.add(Dense(1))
    critic.add(Activation('linear'))

    c = Input(name='dummy', tensor=K.constant(np.array([[1.0]])), shape=(1,))
    test_sigma = Dense(2, use_bias=False)
    out_s = test_sigma(c)

    actor_in = Input(name='actual', shape=(3, 9))
    f_ac_in = Flatten()(actor_in)

    x = Dense(32, activation='relu')(f_ac_in)
    x = Dense(64, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    ac_out = Dense(2, activation='linear')(x)

    actor = Model(inputs=[c, actor_in], outputs=[ac_out, out_s])

    sampler = IndependentGaussianProcess(2)
    memory = SequentialMemory(limit=10, window_length=3)

    agent = PPOAgent(actor, 1, critic, memory, sampler)
    agent.compile('sgd')
    agent.fit(env, nb_steps=20)

#if __name__ == "__main__":
#    test_ppo_basic()
