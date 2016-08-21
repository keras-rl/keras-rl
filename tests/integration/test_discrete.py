
import numpy as np
from numpy.testing import assert_allclose
from gym.envs.debugging.one_round_deterministic_reward import OneRoundDeterministicRewardEnv

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory


def test_dqn():
    env = OneRoundDeterministicRewardEnv()
    np.random.seed(123)
    env.seed(123)
    nb_actions = env.action_space.n

    # Next, we build a very simple model.
    model = Sequential()
    model.add(Dense(16, input_shape=(1,)))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))

    memory = SequentialMemory(limit=1000)
    policy = EpsGreedyQPolicy(eps=.05)
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
                   target_model_update=1e-1, policy=policy, enable_double_dqn=False)
    dqn.compile(Adam(lr=1e-3))

    dqn.fit(env, nb_steps=2000, visualize=False, verbose=0)
    h = dqn.test(env, nb_episodes=20, visualize=False)
    assert_allclose(np.mean(h.history['episode_reward']), 1.)


def test_double_dqn():
    env = OneRoundDeterministicRewardEnv()
    np.random.seed(123)
    env.seed(123)
    nb_actions = env.action_space.n

    # Next, we build a very simple model.
    model = Sequential()
    model.add(Dense(16, input_shape=(1,)))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))

    memory = SequentialMemory(limit=1000)
    policy = EpsGreedyQPolicy(eps=.05)
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
                   target_model_update=1e-1, policy=policy, enable_double_dqn=True)
    dqn.compile(Adam(lr=1e-3))

    dqn.fit(env, nb_steps=2000, visualize=False, verbose=0)
    h = dqn.test(env, nb_episodes=20, visualize=False)
    assert_allclose(np.mean(h.history['episode_reward']), 1.)
