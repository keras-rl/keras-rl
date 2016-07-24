from collections import deque
from copy import deepcopy

import numpy as np
import keras.backend as K

from rl.core import Agent
from rl.policy import EpsGreedyQPolicy
from rl.util import *


def mean_q(y_true, y_pred):
	return K.mean(K.max(y_pred, axis=-1))


# An implementation of the DQN agent as described in Mnih (2013) and Mnih (2015).
# http://arxiv.org/pdf/1312.5602.pdf
# http://arxiv.org/abs/1509.06461
class DQNAgent(Agent):
	def __init__(self, model, nb_actions, memory, window_length=1, policy=EpsGreedyQPolicy(),
				 gamma=.99, batch_size=32, nb_steps_warmup=1000, train_interval=1, memory_interval=1,
				 target_model_update=10000, reward_range=(-np.inf, np.inf),
				 delta_range=(-np.inf, np.inf), enable_double_dqn=True,
				 custom_model_objects={}, processor=None):
		# Validate (important) input.
		if hasattr(model.output, '__len__') and len(model.output) > 1:
			raise ValueError('Model "{}" has more than one output. DQN expects a model that has a single output.'.format(model))
		if model.output._keras_shape != (None, nb_actions):
			raise ValueError('Model output "{}" has invalid shape. DQN expects a model that has one dimension for each action, in this case {}.'.format(model.output, nb_actions))
		
		super(DQNAgent, self).__init__()

		# Soft vs hard target model updates.
		if target_model_update < 0:
			raise ValueError('`target_model_update` must be >= 0.')
		elif target_model_update >= 1:
			# Hard update every `target_model_update` steps.
			target_model_update = int(target_model_update)
		else:
			# Soft update with `(1 - target_model_update) * old + target_model_update * new`.
			target_model_update = float(target_model_update)

		# Parameters.
		self.nb_actions = nb_actions
		self.window_length = window_length
		self.gamma = gamma
		self.batch_size = batch_size
		self.nb_steps_warmup = nb_steps_warmup
		self.train_interval = train_interval
		self.memory_interval = memory_interval
		self.target_model_update = target_model_update
		self.reward_range = reward_range
		self.delta_range = delta_range
		self.enable_double_dqn = enable_double_dqn
		self.custom_model_objects = custom_model_objects

		# Related objects.
		self.model = model
		self.memory = memory
		self.policy = policy
		self.policy._set_agent(self)
		self.processor = processor

		# State.
		self.compiled = False
		self.reset_states()

	def compile(self, optimizer, metrics=[]):
		metrics += [mean_q]  # register default metrics

		# We never train the target model, hence we can set the optimizer and loss arbitrarily.
		self.target_model = clone_model(self.model, self.custom_model_objects)
		self.target_model.compile(optimizer='sgd', loss='mse')
		
		# Compile model.
		if self.target_model_update < 1.:
			# We use the `AdditionalUpdatesOptimizer` to efficiently soft-update the target model.
			updates = get_soft_target_model_updates(self.target_model, self.model, self.target_model_update)
			optimizer = AdditionalUpdatesOptimizer(optimizer, updates)
		def clipped_mse(y_true, y_pred):
			delta = K.clip(y_true - y_pred, self.delta_range[0], self.delta_range[1])
			return K.mean(K.square(delta), axis=-1)
		self.model.compile(optimizer=optimizer, loss=clipped_mse, metrics=metrics)
		
		self.compiled = True

	# TODO: implement support for pickle

	def load_weights(self, filepath):
		self.model.load_weights(filepath)
		self.update_target_model_hard()

	def save_weights(self, filepath, overwrite=False):
		self.model.save_weights(filepath, overwrite=overwrite)

	def reset_states(self):
		self.recent_action = None
		self.recent_observations = deque(maxlen=self.window_length)

	def update_target_model_hard(self):
		self.target_model.set_weights(self.model.get_weights())

	def process_state_batch(self, batch):
		batch = np.array(batch)
		if self.processor is None:
			return batch
		return self.processor.process_state_batch(batch)

	def compute_q_values(self, state):
		batch = self.process_state_batch([state])
		q_values = self.model.predict_on_batch(batch).flatten()
		assert q_values.shape == (self.nb_actions,)
		return q_values

	def forward(self, observation):
		if self.processor is not None:
			observation = self.processor.process_observation(observation)

		# Select an action.
		while len(self.recent_observations) < self.recent_observations.maxlen:
			# Not enough data, fill the recent_observations queue with copies of the current input.
			# This allows us to immediately perform a policy action instead of falling back to random
			# actions.
			self.recent_observations.append(np.copy(observation))
		state = np.array(list(self.recent_observations)[1:] + [observation])
		assert len(state) == self.window_length
		q_values = self.compute_q_values(state)
		action = self.policy.select_action(q_values=q_values)

		# Book-keeping.
		self.recent_observations.append(observation)
		self.recent_action = action
		
		return action

	def backward(self, reward, terminal):
		metrics = [np.nan for _ in self.metrics_names]
		if not self.training:
			# We're done here. No need to update the experience memory since we only use the working
			# memory to obtain the state over the most recent observations.
			return metrics

		# Clip the reward to be in reward_range.
		reward = np.clip(reward, self.reward_range[0], self.reward_range[1])

		# Store most recent experience in memory.
		if self.step % self.memory_interval == 0:
			self.memory.append(self.recent_observations[-1], self.recent_action, reward, terminal)
		
		# Train the network on a single stochastic batch.
		if self.step > self.nb_steps_warmup and self.step % self.train_interval == 0:
			experiences = self.memory.sample(self.batch_size, self.window_length)
			assert len(experiences) == self.batch_size
			
			# Start by extracting the necessary parameters (we use a vectorized implementation).
			state0_batch = []
			reward_batch = []
			action_batch = []
			terminal_batch = []
			state1_batch = []
			for e in experiences:
				state0_batch.append(e.state0)
				state1_batch.append(e.state1)
				reward_batch.append(e.reward)
				action_batch.append(e.action)
				terminal_batch.append(0. if e.terminal else 1.)

			# Prepare and validate parameters.
			state0_batch = self.process_state_batch(state0_batch)
			state1_batch = self.process_state_batch(state1_batch)
			terminal_batch = np.array(terminal_batch)
			reward_batch = np.array(reward_batch)
			assert reward_batch.shape == (self.batch_size,)
			assert terminal_batch.shape == reward_batch.shape
			assert len(action_batch) == len(reward_batch)

			# Compute Q values for mini-batch update.
			if self.enable_double_dqn:
				# According to the paper "Deep Reinforcement Learning with Double Q-learning"
				# (van Hasselt et al., 2015), in Double DQN, the online network predicts the actions
				# while the target network is used to estimate the Q value.
				q_values = self.model.predict_on_batch(state1_batch)
				assert q_values.shape == (self.batch_size, self.nb_actions)
				actions = np.argmax(q_values, axis=1)
				assert actions.shape == (self.batch_size,)

				# Now, estimate Q values using the target network but select the values with the
				# highest Q value wrt to the online model (as computed above).
				target_q_values = self.target_model.predict_on_batch(state1_batch)
				assert target_q_values.shape == (self.batch_size, self.nb_actions)
				q_batch = target_q_values[xrange(self.batch_size), actions]
			else:
				# Compute the q_values given state1, and extract the maximum for each sample in the batch.
				# We perform this prediction on the target_model instead of the model for reasons
				# outlined in Mnih (2015). In short: it makes the algorithm more stable.
				target_q_values = self.target_model.predict_on_batch(state1_batch)
				assert target_q_values.shape == (self.batch_size, self.nb_actions)
				q_batch = np.max(target_q_values, axis=1).flatten()
			assert q_batch.shape == (self.batch_size,)

			# Compute the current activations in the output layer given state0. This is hacky
			# since we do this in the training step anyway, but this is currently the simplest
			# way to set the gradients of the non-affected output units to zero.
			targets = self.model.predict_on_batch(state0_batch)
			assert targets.shape == (self.batch_size, self.nb_actions)

			# Compute r_t + gamma * max_a Q(s_t+1, a) and update the target targets accordingly,
			# but only for the affected output units (as given by action_batch).
			discounted_reward_batch = self.gamma * q_batch
			# Set discounted reward to zero for all states that were terminal.
			discounted_reward_batch *= terminal_batch
			assert discounted_reward_batch.shape == reward_batch.shape
			Rs = reward_batch + discounted_reward_batch
			for target, R, action in zip(targets, Rs, action_batch):
				target[action] = R  # update action with estimated accumulated reward
			targets = np.array(targets).astype('float32')

			# Finally, perform a single update on the entire batch.
			metrics = self.model.train_on_batch(state0_batch, targets)
			metrics += self.policy.run_metrics()

		if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
			self.update_target_model_hard()

		return metrics

	@property
	def metrics_names(self):
		return self.model.metrics_names[:] + self.policy.metrics_names[:]
