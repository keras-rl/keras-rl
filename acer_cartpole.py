import numpy as np
import gym
import copy

from keras import backend as K
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
nenvs = 1
nsteps = 20
# sess = tf.Session()
# K.set_session(sess)

# Defining Model
shape = (nenvs * nsteps,) + obs_shape
inp = K.placeholder(shape=shape)
inputs = Input(tensor=inp, name='inputs')
x = Dense(32, activation='relu')(inputs)
x = Dense(16, activation='relu')(x)

# Actor and Critic Model
actor_output = Dense(nb_actions, activation='softmax')(x)
critic_output = Dense(nb_actions, activation='linear')(x)

model = Model(inputs=[inputs], outputs=[critic_output, actor_output])

print(K.backend())
agent = ACERAgent(model, nb_actions, nenvs=nenvs)

# TODO : Check the implementation of Episodic Memory
# memory = EpisodeMemory(env, nsteps=20)

agent.compile('sgd')
# TODO : Add the arguements

# TODO : Add the arguements
# agent.fit(env, 20)

# TODO : Save weights and do a simple test
# TODO : Add tensorflow optimizer