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
from rl.agents.acer.episode_memory import EpisodeMemory
from rl.policy import SoftmaxPolicy
from rl.common.cmd_util import make_gym_env
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

# TODO : Add support for atari
# The current implementation supports simple toy games.

ENV_NAME = 'CartPole-v1'

# Define the number or environments and steps
nenvs = 4
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
opt = Adam(lr=0.00005, clipvalue=10.)

# Currently compile do not support metrics.
agent.compile(opt)

mode = 'train'
if mode == 'train':
    # Okay, now it's time to learn something! We capture the interrupt exception so that training
    # can be prematurely aborted. Notice that you can the built-in Keras callbacks!
    weights_filename = 'acer_{}_weights.h5f'.format(ENV_NAME)
    checkpoint_weights_filename = 'acer_' + ENV_NAME + '_weights_{step}.h5f'
    log_filename = 'acer_{}_log.json'.format(ENV_NAME)
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=1000)]
    callbacks += [FileLogger(log_filename, interval=5000)]
    agent.fit(env, callbacks=callbacks, nb_steps=50000, log_interval=10000)

    # After training is done, we save the final weights one more time.
    agent.save_weights(weights_filename, overwrite=True)

    # Finally, evaluate our algorithm for 10 episodes.
    env = gym.make(ENV_NAME)
    agent.test(env, nb_episodes=10, visualize=False)
elif mode == 'test':
    weights_filename = 'acer_{}_weights.h5f'.format(ENV_NAME)
    # if args.weights:
    #     weights_filename = args.weights
    agent.load_weights(weights_filename)
    env = gym.make(ENV_NAME)
    agent.test(env, nb_episodes=10, visualize=False)
# print (abc.losses)