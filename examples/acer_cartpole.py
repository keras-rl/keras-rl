import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import gym
import copy

from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

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
nenvs = 1
nsteps = 20
# sess = tf.Session()
# K.set_session(sess)

# Defining Model

def model_fn(inp, name='inputs'):
	inps = Input(tensor=inp, name=name)
	x = Dense(32, activation='relu')(inps)
	x = Dense(16, activation='relu')(x)

	# Actor and Critic Model
	actor_output = Dense(nb_actions, activation='softmax')(x)
	critic_output = Dense(nb_actions, activation='linear')(x)

	inputs = [inps]
	outputs = [critic_output, actor_output]
	model = Model(inputs=inputs, outputs=outputs)
	return model, inputs, outputs

policy = SoftmaxPolicy()
agent = ACERAgent(model_fn, nb_actions, obs_shape, policy=policy, nenvs=nenvs)

# TODO : Check the implementation of Episodic Memory
# memory = EpisodeMemory(env, nsteps=20)
agent.compile('sgd')

agent.fit(env, 1000000)