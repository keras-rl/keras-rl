import os
# To use CPU for faster computation
# Remove this, if not needed
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import gym
import copy

from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam, SGD

from rl.agents import ACERAgent
from rl.episode_memory import EpisodeMemory
from rl.policy import SoftmaxPolicy

ENV_NAME = 'CartPole-v0'

env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)

# Action is discrete
nb_actions = env.action_space.n
obs_shape = env.observation_space.shape

# Define the number or environments and steps
nenvs = 1
nsteps = 50

# Defining model function
def model_fn(inp, name='inputs'):
	inps = Input(tensor=inp, name=name)

	# Define your model here.
	# We are sharing the network parameters.
	x = Dense(32, activation='relu')(inps)
	x = Dense(16, activation='relu')(x)

	# Actor and Critic output for the model
	actor_output = Dense(nb_actions, activation='softmax')(x)
	critic_output = Dense(nb_actions, activation='linear')(x)

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
sgd = SGD(lr=0.0005, decay=1e-6, momentum=0.9, nesterov=True)

# Currently compile do not support metrics.
agent.compile(sgd)

agent.fit(env, 1000000)