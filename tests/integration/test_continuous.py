import random

import numpy as np
from numpy.testing import assert_allclose
import gym

from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Flatten, Input, merge
from keras.optimizers import Adam

from rl.agents import ContinuousDQNAgent, DDPGAgent
from rl.random import OrnsteinUhlenbeckProcess
from rl.memory import SequentialMemory


def test_cdqn():
    # TODO: replace this with a simpler environment where we can actually test if it finds a solution
    env = gym.make('Pendulum-v0')
    np.random.seed(123)
    env.seed(123)
    random.seed(123)
    nb_actions = env.action_space.shape[0]

    V_model = Sequential()
    V_model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    V_model.add(Dense(16))
    V_model.add(Activation('relu'))
    V_model.add(Dense(1))
    V_model.add(Activation('linear'))

    mu_model = Sequential()
    mu_model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    mu_model.add(Dense(16))
    mu_model.add(Activation('relu'))
    mu_model.add(Dense(nb_actions))
    mu_model.add(Activation('linear'))
    
    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
    x = merge([action_input, Flatten()(observation_input)], mode='concat')
    x = Dense(16)(x)
    x = Activation('relu')(x)
    x = Dense(((nb_actions * nb_actions + nb_actions) // 2))(x)
    x = Activation('linear')(x)
    L_model = Model(input=[action_input, observation_input], output=x)

    memory = SequentialMemory(limit=1000, window_length=1)
    random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.3, size=nb_actions)
    agent = ContinuousDQNAgent(nb_actions=nb_actions, V_model=V_model, L_model=L_model, mu_model=mu_model,
                               memory=memory, nb_steps_warmup=50, random_process=random_process,
                               gamma=.99, target_model_update=1e-3)
    agent.compile(Adam(lr=1e-3))

    agent.fit(env, nb_steps=400, visualize=False, verbose=0, nb_max_episode_steps=100)
    h = agent.test(env, nb_episodes=2, visualize=False, nb_max_episode_steps=100)
    # TODO: evaluate history


def test_ddpg():
    # TODO: replace this with a simpler environment where we can actually test if it finds a solution
    env = gym.make('Pendulum-v0')
    np.random.seed(123)
    env.seed(123)
    random.seed(123)
    nb_actions = env.action_space.shape[0]

    actor = Sequential()
    actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    actor.add(Dense(16))
    actor.add(Activation('relu'))
    actor.add(Dense(nb_actions))
    actor.add(Activation('linear'))

    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
    flattened_observation = Flatten()(observation_input)
    x = merge([action_input, flattened_observation], mode='concat')
    x = Dense(16)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)
    critic = Model(input=[action_input, observation_input], output=x)
    
    memory = SequentialMemory(limit=1000, window_length=1)
    random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.3)
    agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                      memory=memory, nb_steps_warmup_critic=50, nb_steps_warmup_actor=50,
                      random_process=random_process, gamma=.99, target_model_update=1e-3)
    agent.compile([Adam(lr=1e-3), Adam(lr=1e-3)])

    agent.fit(env, nb_steps=400, visualize=False, verbose=0, nb_max_episode_steps=100)
    h = agent.test(env, nb_episodes=2, visualize=False, nb_max_episode_steps=100)
    # TODO: evaluate history
