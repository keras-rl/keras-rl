import os
# To use CPU for faster computation
# Remove this, if not needed
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import numpy as np
import gym
import copy

from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, ReLU
from keras.optimizers import Adam, SGD

from rl.agents import ACERAgent
from rl.episode_memory import EpisodeMemory
from rl.policy import SoftmaxPolicy
from rl.common.cmd_util import make_gym_env

# TODO : Add support for atari
# The current implementation supports simple toy games.

ENV_NAME = 'CartPole-v1'

# Define the number or environments and steps
nenvs = 2
nsteps = 50

# make_gym_env : Makes synchronous environments
# make_gym_env only supports actor-critic frameworks

env = make_gym_env(ENV_NAME, nenvs, 123)
np.random.seed(123)
env.seed(123)

# Action is discrete
nb_actions = env.action_space.n
obs_shape = env.observation_space.shape

# Defining model function
def model_fn(inp, name='inputs'):
	inps = Input(tensor=inp, name=name)

	# Define your model here.

	# Note : Parameter sharing is not working
	# Hence define two different parallel models
	# for critic and actor networks.

	x_actor = Dense(32, activation='relu')(inps)
	x_actor = Dense(16)(x_actor)
	x_actor = ReLU(max_value=80.)(x_actor)

	x_critic = Dense(32, activation='relu')(inps)
	x_critic = Dense(16, activation='relu')(x_critic)

	# Actor and Critic output for the model
	actor_output = Dense(nb_actions, activation='softmax')(x_actor)
	critic_output = Dense(nb_actions, activation='linear')(x_critic)

	# Input list to the model
	inputs = [inps]

	# Output list to the model
	outputs = [critic_output, actor_output]

	model = Model(inputs=inputs, outputs=outputs)
	return model, inputs, outputs

# Policy of the actor model.
policy = SoftmaxPolicy()

# Experience memory of the agent
memory = EpisodeMemory(nsteps, 50000)
agent = ACERAgent(memory, model_fn, nb_actions, obs_shape, policy=policy, nenvs=nenvs, nsteps=nsteps)

# Define the optimizor to be used
sgd = Adam(lr=0.00005, clipvalue=10.)

# Currently compile do not support metrics.
agent.compile(sgd)

agent.fit(env, 1000000)
