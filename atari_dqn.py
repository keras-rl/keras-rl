from argparse import ArgumentParser

from PIL import Image
import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint

from rl.agents import DQNAgent
from rl.memory import Memory
from rl.core import Processor
from rl.callbacks import FileLogger


INPUT_SHAPE = (84, 84)
ENV_NAME = 'Breakout-v0'
WINDOW_LENGTH = 4


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

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = Memory(limit=1000000)
processor = AtariProcessor()
dqn = DQNAgent(model=model, nb_actions=nb_actions, window_length=WINDOW_LENGTH, memory=memory,
	processor=processor, nb_steps_warmup=50000, gamma=.99, train_interval=1, delta_range=(-1., 1.))
dqn.compile(RMSprop(lr=.00025, clipvalue=10.), metrics=['mae'])

# Okay, now it's time to learn something! We capture the interrupt exception so that training
# can be prematurely aborted. Notice that you can the built-in Keras callbacks!
weights_filename = 'dqn_{}_weights.h5f'.format(ENV_NAME)
log_filename = 'dqn_{}_log.json'.format(ENV_NAME)
callbacks = [ModelCheckpoint(weights_filename), FileLogger(log_filename)]
try:
	dqn.fit(env, callbacks=callbacks, nb_steps=10000000, action_repetition=4,
		nb_max_random_start_steps=30, log_interval=10000)
except KeyboardInterrupt:
	# Ignore this, and continue with the rest.
	pass

# After training is done, we save the final weights one more time.
dqn.save_weights(weights_filename, overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, action_repetition=4)
