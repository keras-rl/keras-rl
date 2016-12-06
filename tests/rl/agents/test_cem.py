from __future__ import division
from __future__ import absolute_import

import pytest
import numpy as np
from numpy.testing import assert_allclose

from keras.models import Model, Sequential
from keras.layers import Input, merge, Dense, Flatten

from rl.agents.cem import CEMAgent
from rl.memory import EpisodeParameterMemory
from rl.core import MultiInputProcessor

from ..util import MultiInputTestEnv


def test_single_cem_input():
    model = Sequential()
    model.add(Flatten(input_shape=(2, 3)))
    model.add(Dense(2))

    memory = EpisodeParameterMemory(limit=10, window_length=2)
    agent = CEMAgent(model, memory=memory, nb_actions=2, nb_steps_warmup=5, batch_size=4, train_interval=50)
    agent.compile()
    agent.fit(MultiInputTestEnv((3,)), nb_steps=100)


def test_multi_cem_input():
    input1 = Input(shape=(2, 3))
    input2 = Input(shape=(2, 4))
    x = merge([input1, input2], mode='concat')
    x = Flatten()(x)
    x = Dense(2)(x)
    model = Model(input=[input1, input2], output=x)

    memory = EpisodeParameterMemory(limit=10, window_length=2)
    processor = MultiInputProcessor(nb_inputs=2)
    agent = CEMAgent(model, memory=memory, nb_actions=2, nb_steps_warmup=5, batch_size=4,
                     processor=processor, train_interval=50)
    agent.compile()
    agent.fit(MultiInputTestEnv([(3,), (4,)]), nb_steps=100)
