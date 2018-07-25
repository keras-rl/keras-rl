import numpy as np
import gym
import copy

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from rl.agents import ACERAgent
from rl.episode_memory import EpisodeMemory

ENV_NAME = 'CartPole-v0'

env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)

# Action is discrete
nb_actions = env.action_space.n
obs_shape = env.observation_space.shape

# Defining Model
inputs = Input(shape=obs_shape, name='inputs')

x = Dense(32, activation='relu')(inputs)
x = Dense(32, activation='relu')(x)

# Actor and Critic Model
actor_output = Dense(nb_actions, activation='softmax')(x)
critic_output = Dense(nb_actions)(x)

actor = Model(inputs=[inputs], outputs=actor_output)
critic = Model(inputs=[inputs], outputs=critic_output)

avg_actor_model = copy.deepcopy(actor)
avg_critic_model = copy.deepcopy(critic)

agent = ACERAgent(actor, critic, avg_actor_model, avg_critic_model)

# TODO : Check the implementation of Episodic Memory
memory = EpisodeMemory(env, nsteps=20)

# TODO : Add the arguements
agent = ACERAgent()

# TODO : Add the arguements
agent.fit()

# TODO : Save weights and do a simple test