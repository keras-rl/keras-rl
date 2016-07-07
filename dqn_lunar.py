from argparse import ArgumentParser

import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Nadam
from keras.callbacks import ModelCheckpoint

from rl.agents import DQNAgent
from rl.agents.dqn import DQNAgent, AnnealedQPolicy
from rl.memory import Memory
from rl.callbacks import FileLogger


ENV_NAME = 'LunarLander-v2'
WINDOW_LENGTH = 4


# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
nb_actions = env.action_space.n

# Next, we build our model. We use the same model that was described by Mnih et al. (2015).
model = Sequential()
model.add(Flatten(input_shape=(WINDOW_LENGTH,) + env.observation_space.shape))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = Memory(limit=100000)
policy = AnnealedQPolicy(nb_annealing_steps=100000)
dqn = DQNAgent(model=model, policy=policy, nb_actions=nb_actions, window_length=WINDOW_LENGTH, memory=memory,
	nb_steps_warmup=5000, gamma=.99, train_interval=1, delta_range=(-1., 1.))
dqn.compile(Nadam(lr=.00025), metrics=['mae'])

# Okay, now it's time to learn something! We capture the interrupt exception so that training
# can be prematurely aborted. Notice that you can the built-in Keras callbacks!
weights_filename = 'dqn_{}_weights.h5f'.format(ENV_NAME)
log_filename = 'dqn_{}_log.json'.format(ENV_NAME)
callbacks = [ModelCheckpoint(weights_filename)]
callbacks += [FileLogger(log_filename, save_continiously=True)]
dqn.fit(env, callbacks=callbacks, nb_steps=1000000, action_repetition=1,
		nb_max_random_start_steps=10, log_interval=10000, visualize=True)

# After training is done, we save the final weights one more time.
dqn.save_weights(weights_filename, overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, action_repetition=1, visualize=False)
