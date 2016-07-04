from argparse import ArgumentParser

import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D
from keras.optimizers import RMSprop

from rl.agents import DQNAgent
from rl.memory import Memory
from rl.core import Processor

import PIL
from PIL import Image


INPUT_SHAPE = (84, 84)
ENV_NAME = 'Breakout-v0'
WINDOW_LENGTH = 4


def rgb2gray(rgb):
	# https://en.wikipedia.org/wiki/Grayscale#Converting_color_to_grayscale
    return np.dot(rgb, [0.299, 0.587, 0.114])


class AtariProcessor(Processor):
	def process_observation(self, observation):
		assert observation.ndim == 3  # (height, width, channel)
		img = Image.fromarray(observation)
		img = img.resize(INPUT_SHAPE).convert('L')  # resize and convert to grayscale
		processed_observation = np.array(img)
		assert processed_observation.shape == INPUT_SHAPE
		return processed_observation.astype('uint8')  # saves storage in experience memory

	def process_state_batch(self, batch):
		# We could perform this processing step in `process_observation`. In this case, however,
		# we would need to store a `float32` array instead, which is 4x more memory intensive than
		# an `uint8` array. This matters if we store 1M observations.
		processed_batch = batch.astype('float32') / 255.
		return processed_batch


# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
nb_actions = env.action_space.n

# Next, we build our model. We use the same model that was described by Mnih et al. (2015).
model = Sequential()
model.add(Convolution2D(32, 8, 8, subsample=(4, 4), input_shape=(WINDOW_LENGTH,) + INPUT_SHAPE))
model.add(Activation('relu'))
model.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, subsample=(1, 1)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# Finally, we configure our agent.
memory = Memory(limit=1000000)
processor = AtariProcessor()
dqn = DQNAgent(model=model, nb_actions=nb_actions, window_length=WINDOW_LENGTH, memory=memory,
	processor=processor)
dqn.compile(RMSprop(lr=.00025))
dqn.fit(env)
