from collections import deque
from copy import deepcopy

import numpy as np
from keras.models import model_from_config, Sequential, Graph

from rl.core import Agent


# TODO: move into util
def clone_model(model, custom_objects={}):
	config = model.get_config()
	clone = Sequential.from_config(config, custom_objects)
	clone.set_weights(model.get_weights())
	return clone


# TODO: do we still need this in Keras 1.x?
def predict_on_batch(model, input_batch, output_name='output'):
	output = None
	if isinstance(model, Graph):
		output = model.predict_on_batch(input_batch)[output_name]
	elif isinstance(model, Sequential):
		output = model.predict_on_batch(input_batch)
	else:
		raise RuntimeError('unknown model type')
	return output


# TODO: do we still need this in Keras 1.x?
def train_on_batch(model, input_batch, target_batch, output_name='output'):
	metrics = None
	if isinstance(model, Graph):
		data = dict(input_batch)  # shallow copy
		data[output_name] = target_batch
		metrics = model.train_on_batch(data)
	elif isinstance(model, Sequential):
		input_batch = np.array(input_batch)
		target_batch = np.array(target_batch)
		assert input_batch.shape[0] == target_batch.shape[0]
		metrics = model.train_on_batch(input_batch, target_batch)
	else:
		raise RuntimeError('unknown model type')
	assert metrics is not None

	if not isinstance(metrics, list):
		metrics = [metrics]
	return metrics


class QPolicy(object):
	def select_action(self, q_values, step=1, training=True):
		raise NotImplementedError()

	@property
	def metrics_names(self):
		return []

	def get_metrics(self, step=1, training=True):
		return []


class AnnealedQPolicy(QPolicy):
	def __init__(self, eps_max=1., eps_min=.1, eps_test=.05, nb_annealing_steps=500000):
		super(AnnealedQPolicy, self).__init__()
		self.eps_max = eps_max
		self.eps_min = eps_min
		self.eps_test = eps_test
		self.nb_annealing_steps = nb_annealing_steps

	def compute_eps(self, step, training):
		if training:
			m = -float(self.eps_max - self.eps_min) / float(self.nb_annealing_steps)
			c = float(self.eps_max)
			eps = max(self.eps_min, m * float(step) + c)
		else:
			eps = self.eps_test
		return eps

	def select_action(self, q_values, step=1, training=True):
		assert q_values.ndim == 1
		nb_actions = q_values.shape[0]

		eps = self.compute_eps(step, training)
		if np.random.uniform() < eps:
			action = np.random.random_integers(0, nb_actions-1)
		else:
			action = np.argmax(q_values)
		return action

	@property
	def metrics_names(self):
		return ['mean_eps']

	def get_metrics(self, step=1, training=True):
		return [self.compute_eps(step, training)]


class BoltzmannQPolicy(QPolicy):
	def __init__(self, temperature=1.):
		super(BoltzmannQPolicy, self).__init__()
		self.temperature = temperature

	def select_action(self, q_values, step=1, training=True):
		assert q_values.ndim == 1
		nb_actions = q_values.shape[0]

		exp_values = np.exp(q_values / self.temperature)
		probs = exp_values / np.sum(exp_values)
		action = np.random.choice(range(nb_actions), p=probs)
		return action


# An implementation of the DQN agent as described in Mnih (2013) and Mnih (2015).
# http://arxiv.org/pdf/1312.5602.pdf
# http://arxiv.org/abs/1509.06461
class DQNAgent(Agent):
	def __init__(self, model, nb_actions, memory, window_length, policy=AnnealedQPolicy(),
				 gamma=.9, batch_size=32, nb_steps_warmup=1000, train_interval=4, memory_interval=1,
				 target_model_update_interval=10000, reward_range=(-np.inf, np.inf),
				 delta_range=(-np.inf, np.inf), enable_double_dqn=True,
				 custom_model_objects={}, processor=None):
		super(DQNAgent, self).__init__()

		self.model = model
		self.target_model = clone_model(self.model, custom_model_objects)
		self.nb_actions = nb_actions
		self.window_length = window_length

		# Parameters.
		self.gamma = gamma
		self.batch_size = batch_size
		self.nb_steps_warmup = nb_steps_warmup
		self.train_interval = train_interval
		self.memory_interval = memory_interval
		self.target_model_update_interval = target_model_update_interval
		self.reward_range = reward_range
		self.delta_range = delta_range
		self.enable_double_dqn = enable_double_dqn
		self.custom_model_objects = custom_model_objects

		# Related objects.
		self.memory = memory
		self.policy = policy
		self.processor = processor

		# State.
		self.compiled = False
		self.step = 0
		self.reset_states()

	def compile(self, optimizer, metrics=[]):
		self.compiled = True
		self.model.compile(optimizer=optimizer, loss='mse', metrics=metrics)
		self.target_model.compile(optimizer=optimizer, loss='mse', metrics=metrics)

	# TODO: implement support for pickle

	def load_weights(self, filepath):
		self.model.load_weights(filepath)
		self.update_target_model()

	def save_weights(self, filepath, overwrite=False):
		self.model.save_weights(filepath, overwrite=overwrite)

	def reset_states(self):
		self.recent_action = None
		self.recent_observations = deque(maxlen=self.window_length)

	def update_target_model(self):
		self.target_model.set_weights(self.model.get_weights())

	def process_state_batch(self, batch):
		batch = np.array(batch)
		if self.processor is None:
			return batch
		return self.processor.process_state_batch(batch)

	def compute_q_values(self, state):
		batch = self.process_state_batch([state])
		q_values = predict_on_batch(self.model, batch).flatten()
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
		action = self.policy.select_action(q_values, step=self.step, training=self.training)

		# Book-keeping.
		self.recent_observations.append(observation)
		self.recent_action = action
		
		return action

	def backward(self, reward, terminal):
		self.step += 1
		metrics = [np.nan for _ in self.metrics_names]
		if not self.training:
			# We're done here. No need to update the experience memory since we only use the working
			# memory to obtain the state over the most recent observations.
			return metrics

		# Clip the reward to be in reward_range.
		reward = min(max(reward, self.reward_range[0]), self.reward_range[1])

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

			# Prepare and validate parameters
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
				q_values = predict_on_batch(self.model, state1_batch)
				assert q_values.shape == (self.batch_size, self.nb_actions)
				actions = np.argmax(q_values, axis=1)
				assert actions.shape == (self.batch_size,)

				# Now, estimate Q values using the target network but select the values with the
				# highest Q value wrt to the online model (as computed above).
				target_q_values = predict_on_batch(self.target_model, state1_batch)
				assert target_q_values.shape == (self.batch_size, self.nb_actions)
				q_batch = target_q_values[xrange(self.batch_size), actions]
			else:
				# Compute the q_values given state1, and extract the maximum for each sample in the batch.
				# We perform this prediction on the target_model instead of the model for reasons
				# outlined in Mnih (2015). In short: it makes the algorithm more stable.
				target_q_values = predict_on_batch(self.target_model, state1_batch)
				assert target_q_values.shape == (self.batch_size, self.nb_actions)
				q_batch = np.max(target_q_values, axis=1).flatten()
			assert q_batch.shape == (self.batch_size,)

			# Compute the current activations in the output layer given state0. This is hacky
			# since we do this in the training step anyway, but this is currently the simplest
			# way to set the gradients of the non-affected output units to zero.
			ys = predict_on_batch(self.model, state0_batch)
			mean_q = np.mean(np.max(ys, axis=-1))
			assert ys.shape == (self.batch_size, self.nb_actions)

			# Compute r_t + gamma * max_a Q(s_t+1, a) and update the target ys accordingly,
			# but only for the affected output units (as given by action_batch).
			discounted_reward_batch = self.gamma * q_batch
			# Set discounted reward to zero for all states that were terminal.
			discounted_reward_batch *= terminal_batch
			assert discounted_reward_batch.shape == reward_batch.shape
			rs = reward_batch + discounted_reward_batch
			for y, r, action in zip(ys, rs, action_batch):
				# This might seem confusing at first. What happens here is simply the following
				# equation: y[action] = r iff delta_range=(-np.inf, np.inf) since in this case
				# delta = r - y[action] and y[action] = y[action] + delta = y[action] + r - y[action] = r.
				# The catch, however, is that we can now clip the delta to a certain range, e.g. (-1, 1).
				delta = r - y[action]
				y[action] = y[action] + np.clip(delta, self.delta_range[0], self.delta_range[1])
			ys = np.array(ys).astype('float32')

			# Finally, perform a single update on the entire batch.
			metrics = train_on_batch(self.model, state0_batch, ys)
			metrics.append(mean_q)
			metrics.extend(self.policy.get_metrics(step=self.step, training=self.training))

		if self.step % self.target_model_update_interval == 0:
			self.update_target_model()

		return metrics

	@property
	def metrics_names(self):
		return self.model.metrics_names[:] + ['mean_q'] + self.policy.metrics_names[:]
