from PIL import Image
import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D
from keras.optimizers import Nadam
from keras.callbacks import ModelCheckpoint

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
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
memory = SequentialMemory(limit=1000000)
processor = AtariProcessor()

# Select a policy. We use eps-greedy action selection, which means that a random action is selected
# with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
# the agent initially explores the environment (high eps) and then gradually sticks to what it knows
# (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05
# so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
	nb_steps=1000000)

# The trade-off between exploration and exploitation is difficult and an on-going research topic.
# If you want, you can experiment with the parameters or use a different policy. Another popular one
# is Boltzmann-style exploration:
# policy = BoltzmannQPolicy(tau=1.)
# Feel free to give it a try!

dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, window_length=WINDOW_LENGTH, memory=memory,
	processor=processor, nb_steps_warmup=50000, gamma=.99, train_interval=4, delta_range=(-10., 10.))
dqn.compile(Nadam(lr=.00025), metrics=['mae'])

# Okay, now it's time to learn something! We capture the interrupt exception so that training
# can be prematurely aborted. Notice that you can the built-in Keras callbacks!
weights_filename = 'dqn_{}_weights.h5f'.format(ENV_NAME)
log_filename = 'dqn_{}_log.json'.format(ENV_NAME)
callbacks = [ModelCheckpoint(weights_filename)]
callbacks += [FileLogger(log_filename, save_continiously=False)]
dqn.fit(env, callbacks=callbacks, nb_steps=20000000, nb_max_random_start_steps=10, log_interval=10000)

# After training is done, we save the final weights one more time.
dqn.save_weights(weights_filename, overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=10)
