from collections import deque
from copy import deepcopy, copy
import os
import threading
from Queue import Queue

import numpy as np
from keras.models import model_from_config, Sequential, Graph
from keras.utils.layer_utils import container_from_config
from keras import optimizers
from keras import backend as K
import keras.callbacks as cbks
from keras import objectives

from rl.memory import Memory
from rl.random import OrnsteinUhlenbeckProcess
from rl.callbacks import TestLogger, TrainEpochLogger, TrainIntervalLogger


def collect_non_trainable_weights(model):
	weights = []
	if hasattr(model, 'non_trainable_weights'):
		weights += model.non_trainable_weights
	if hasattr(model, 'nodes'):  # Graph
		for n in model.nodes.values():
			weights += collect_non_trainable_weights(n)
	if hasattr(model, 'layers'):  # Sequential
		for l in model.layers:
			weights += collect_non_trainable_weights(l)
	return weights


def get_target_model_updates(target, source, tau):
	target_params = target.trainable_weights + collect_non_trainable_weights(target)
	source_params = source.trainable_weights + collect_non_trainable_weights(source)
	assert len(target_params) == len(source_params)

	# Create updates.
	updates = []
	for tp, sp in zip(target_params, source_params):
		updates.append((tp, tau * sp + (1. - tau) * tp))
	return updates


class AdditionalUpdatesOptimizer(optimizers.Optimizer):
	def __init__(self, optimizer, additional_updates):
		super(AdditionalUpdatesOptimizer, self).__init__()
		self.optimizer = optimizer
		self.additional_updates = additional_updates

	def get_updates(self, params, constraints, loss):
		updates = self.optimizer.get_updates(params, constraints, loss)
		updates += self.additional_updates
		self.updates = updates
		return self.updates

	def get_config(self):
		return self.optimizer.get_config()


def clone_optimizer(optimizer):
	params = dict([(k, v) for k, v in optimizer.get_config().items()])
	name = params.pop('name')
	clone = optimizers.get(name, params)
	if hasattr(optimizer, 'clipnorm'):
		clone.clipnorm = optimizer.clipnorm
	if hasattr(optimizer, 'clipvalue'):
		clone.clipvalue = optimizer.clipvalue
	return clone


def clone_model(model, custom_objects={}):
	config = model.get_config()
	clone = model_from_config(config, custom_objects)  # compiles the model internally
	clone.set_weights(model.get_weights())
	return clone


def clone_container(container, custom_objects={}):
	config = container.get_config()
	clone = container_from_config(config, custom_objects)  # containers cannot be compiled
	clone.set_weights(container.get_weights())
	return clone


def predict_on_batch(model, input_batch, output_name='output'):
	output = None
	n_samples = None
	if isinstance(model, Graph):
		n_samples = len(input_batch[input_batch.keys()[0]])
		output = model.predict_on_batch(input_batch)[output_name]
	elif isinstance(model, Sequential):
		n_samples = len(input_batch)
		input_batch = np.array(input_batch)
		output = model.predict_on_batch(input_batch)[0]  # this is weird, but for some reason the output is always contained in a single-element array
	else:
		raise RuntimeError('unknown model type')
	return output


def train_on_batch(model, input_batch, target_batch, output_name='output'):
	loss = 0.
	if isinstance(model, Graph):
		data = dict(input_batch)  # shallow copy
		data[output_name] = target_batch
		loss = model.train_on_batch(data)[0]
	elif isinstance(model, Sequential):
		input_batch = np.array(input_batch)
		target_batch = np.array(target_batch)
		assert input_batch.shape[0] == target_batch.shape[0]
		loss = model.train_on_batch(input_batch, target_batch)[0]
	else:
		raise RuntimeError('unknown model type')
	return loss


class Agent(object):
	def fit(self, game, nb_epoch=100, action_repetition=1, callbacks=[], verbose=1):
		if action_repetition < 1:
			raise ValueError('action_repetition must be >= 1')

		self.training = True
		game.training = True

		if verbose > 2:
			callbacks += [TrainEpochLogger()]
		elif verbose == 1:
			callbacks += [TrainIntervalLogger()]
		callbacks = cbks.CallbackList(callbacks)
		callbacks._set_model(self)
		# TODO: extend this
		callbacks._set_params({
			'nb_epoch': nb_epoch,
		})
		callbacks.on_train_begin()

		for epoch_idx in xrange(nb_epoch):
			callbacks.on_epoch_begin(epoch_idx)
			batch_idx = 0
			total_reward = 0.
			while not game.game_over:
				callbacks.on_batch_begin(batch_idx)
				
				# This is were all of the work happens. We first perceive and compute the action
				# (forward step) and then use the reward to improve (backward step).
				ins = game.perceive()
				action = self.forward(ins)
				reward = 0.
				for _ in xrange(action_repetition):
					reward += game.act(action)
					if game.game_over:
						break
				stats = self.backward(reward, terminal=game.game_over)
				total_reward += reward
				
				batch_logs = {
					'action': action,
					'input': ins,
					'reward': reward,
					'stats': stats,
					'epoch': epoch_idx,
				}
				callbacks.on_batch_end(batch_idx, batch_logs)
				batch_idx += 1
			epoch_logs = {
				'total_reward': total_reward,
			}
			callbacks.on_epoch_end(epoch_idx, epoch_logs)
			game.reset()
			self.reset()
		callbacks.on_train_end()

	def test(self, game, nb_epoch=1, action_repetition=1, callbacks=[]):
		# TODO: implement callbacks. The Keras callbacks do not really fit, maybe we need to extend.
		# TODO: implement visualization support
		# TODO: implement test logger
		if action_repetition < 1:
			raise ValueError('action_repetition must be >= 1')

		self.training = False
		game.training = False

		callbacks += [TestLogger()]
		callbacks = cbks.CallbackList(callbacks)
		callbacks._set_model(self)
		callbacks._set_params({
			'nb_epoch': nb_epoch,
		})

		for epoch_idx in xrange(nb_epoch):
			callbacks.on_epoch_begin(epoch_idx)
			total_reward = 0.
			batch_idx = 0
			while not game.game_over:
				callbacks.on_batch_begin(batch_idx)

				ins = game.perceive()
				action = self.forward(ins)
				reward = 0.
				for _ in xrange(action_repetition):
					reward += game.act(action)
					if game.game_over:
						break
				self.backward(reward, terminal=game.game_over)
				total_reward += reward
				
				callbacks.on_batch_end(batch_idx)
				batch_idx += 1
			epoch_logs = {
				'total_reward': total_reward,
			}
			callbacks.on_epoch_end(epoch_idx, epoch_logs)
			game.reset()
			self.reset()

	def reset(self):
		pass

	def forward(self, current_input):
		raise NotImplementedError()

	def backward(self, reward, terminal=False):
		raise NotImplementedError()

	def load_weights(self, filepath):
		raise NotImplementedError()

	def save_weights(self, filepath, overwrite=False):
		raise NotImplementedError()


# An implementation of the DQN agent as described in Mnih (2013) and Mnih (2015).
# http://arxiv.org/pdf/1312.5602.pdf
# http://arxiv.org/abs/1509.06461
class DQNAgent(Agent):
	def __init__(self, n_actions, model, temporal_window=4,
				 eps_start=1., eps_min=.05, eps_test=.01, gamma=.9, batch_size=32,
				 memory_limit=100000, n_steps_warmup=1000, n_steps_eps_annealing=10000,
				 train_interval=4, memory_interval=1, reward_range=(-np.inf, np.inf),
				 delta_range=(-np.inf, np.inf), target_model_update_interval=10000,
				 input_batch_processor=None, enable_terminal=True, enable_double_dqn=True,
				 action_selection='eps-greedy', custom_objects={}):
		self.n_actions = n_actions
		self.model = model
		self.target_model = clone_model(self.model, custom_objects)
		self.input_batch_processor = input_batch_processor

		self.eps_start = eps_start
		self.eps = eps_start
		self.eps_min = eps_min
		self.eps_test = eps_test
		self.action_selection = action_selection
		self.action_selection_func = self.get_action_selection_func(action_selection)
		assert self.action_selection is not None
		
		self.n_steps = 0  # the number of performed steps
		self.n_steps_warmup = n_steps_warmup  # the number of steps before eps annealing and learning start
		self.n_steps_eps_annealing = n_steps_eps_annealing  # the number of total steps to perform before eps_min will be reached
		self.training = False

		# Training parameters
		self.reward_range = reward_range
		self.delta_range = delta_range
		self.gamma = gamma
		self.batch_size = batch_size
		self.train_interval = train_interval
		self.memory_interval = memory_interval
		self.target_model_update_interval = target_model_update_interval
		self.enable_terminal = enable_terminal
		self.enable_double_dqn = enable_double_dqn

		# Book-keeping
		self.temporal_window = temporal_window
		self.memory_limit = memory_limit
		self.memory = Memory(temporal_window=temporal_window, memory_limit=memory_limit)
		self.reset()

	def __getstate__(self):
		d = self.__dict__.copy()

		# Remove things we cannot serialize.
		del d['action_selection_func']

		# Serialize model and target model manually.
		d['model'] = self.model.get_config()
		d['target_model'] = self.target_model.get_config()

		return d

	def __setstate__(self, d):
		# Re-create models from serialized configuration.
		model = model_from_config(d['model'])
		target_model = model_from_config(d['target_model'])

		# Update remaining properties.
		del d['model']
		del d['target_model']
		self.__dict__.update(d)
		self.action_selection_func = self.get_action_selection_func(self.action_selection)
		assert self.action_selection is not None

		# Use models. Weights need to be loaded externally.
		self.model = model
		self.target_model = target_model

	def load_weights(self, filepath):
		self.model.load_weights(filepath)
		self.update_target_model()

	def save_weights(self, filepath, overwrite=False):
		self.model.save_weights(filepath, overwrite=overwrite)

	def get_action_selection_func(self, action_selection):
		if action_selection == 'eps-greedy':
			return self.eps_greedy_action
		elif action_selection == 'boltzmann':
			return self.boltzmann_action
		else:
			return None

	def reset(self):
		self.recent_inputs = deque(maxlen=self.temporal_window)
		self.recent_action = None

	def update_target_model(self):
		self.target_model.set_weights(self.model.get_weights())

	def process_input_batch(self, input_batch):
		if self.input_batch_processor is None:
			return input_batch
		processed_input_batch = self.input_batch_processor(input_batch)
		return processed_input_batch

	def compute_q_values(self, state):
		batch = self.process_input_batch([state])
		q_values = predict_on_batch(self.model, batch).flatten()
		assert q_values.shape == (self.n_actions,)
		return q_values

	def boltzmann_action(self, state):
		q_values = self.compute_q_values(state)
		if self.eps == 0.:
			# Avoid division by zero.
			return np.argmax(q_values)

		# Compute probabilities and sample.
		exp_values = np.exp(q_values / self.eps)
		probs = exp_values / np.sum(exp_values)
		action = np.random.choice(range(self.n_actions), p=probs)
		return action

	def eps_greedy_action(self, state):
		if np.random.uniform() < self.eps:
			action = np.random.random_integers(0, self.n_actions - 1)
		else:
			q_values = self.compute_q_values(state)
			action = np.argmax(q_values)
		return action

	def forward(self, current_input):
		# Compute eps value using annealing to decide if we take a random or policy action.
		if self.training:
			n_steps_train = max(0, self.n_steps - self.n_steps_warmup)
			m = -float(self.eps_start - self.eps_min) / float(self.n_steps_eps_annealing)
			c = float(self.eps_start)
			self.eps = max(self.eps_min, m * float(n_steps_train) + c)
		else:
			self.eps = self.eps_test

		# Select an action.
		while len(self.recent_inputs) < self.recent_inputs.maxlen:
			# Not enough data, fill the recent_inputs array with
			# the current input. This allows us to immediately
			# perform a policy action instead of falling back
			# to random actions.
			self.recent_inputs.append(deepcopy(current_input))
		state = list(self.recent_inputs)[1:] + [current_input]
		assert len(state) == self.temporal_window
		action = self.action_selection_func(state)

		# Book-keeping.
		self.recent_inputs.append(current_input)
		self.recent_action = action
		
		return action

	def backward(self, reward, terminal=False):
		self.n_steps += 1
		if not self.training:
			# We're done here
			return 0.

		# Clip the reward to be in reward_range.
		reward = min(max(reward, self.reward_range[0]), self.reward_range[1])

		# Store most recent experience in memory.
		if self.n_steps % self.memory_interval == 0:
			self.memory.append(self.recent_inputs[-1], self.recent_action, reward, terminal)
		
		# Train the network on a single stochastic batch.
		if self.n_steps > self.n_steps_warmup and self.n_steps % self.train_interval == 0:
			experiences = self.memory.sample(self.batch_size)
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
			state0_batch = self.process_input_batch(state0_batch)
			state1_batch = self.process_input_batch(state1_batch)
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
				assert q_values.shape == (self.batch_size, self.n_actions)
				actions = np.argmax(q_values, axis=1)
				assert actions.shape == (self.batch_size,)

				# Now, estimate Q values using the target network but select the values with the
				# highest Q value wrt to the online model (as computed above).
				target_q_values = predict_on_batch(self.target_model, state1_batch)
				assert target_q_values.shape == (self.batch_size, self.n_actions)
				q_batch = target_q_values[xrange(self.batch_size), actions]
			else:
				# Compute the q_values given state1, and extract the maximum for each sample in the batch.
				# We perform this prediction on the target_model instead of the model for reasons
				# outlined in Mnih (2015). In short: it makes the algorithm more stable.
				target_q_values = predict_on_batch(self.target_model, state1_batch)
				assert target_q_values.shape == (self.batch_size, self.n_actions)
				q_batch = np.max(target_q_values, axis=1).flatten()
			assert q_batch.shape == (self.batch_size,)

			# Compute the current activations in the output layer given state0. This is hacky
			# since we do this in the training step anyway, but this is currently the simplest
			# way to set the gradients of the non-affected output units to zero.
			ys = predict_on_batch(self.model, state0_batch)
			assert ys.shape == (self.batch_size, self.n_actions)

			# Compute r_t + gamma * max_a Q(s_t+1, a) and update the target ys accordingly,
			# but only for the affected output units (as given by action_batch).
			discounted_reward_batch = self.gamma * q_batch
			if self.enable_terminal:
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
			loss = train_on_batch(self.model, state0_batch, ys)
		else:
			loss = 0.

		if self.n_steps % self.target_model_update_interval == 0:
			self.update_target_model()

		return loss


# Deep DPG as described by Lillicrap et al. (2015)
# http://arxiv.org/pdf/1509.02971v2.pdf
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.646.4324&rep=rep1&type=pdf
class DDPGAgent(Agent):
	def __init__(self, n_actions, actor, critic, temporal_window=4,
				 gamma=.9, batch_size=32, memory_limit=100000, n_steps_actor_warmup=1000,
				 n_steps_critic_warmup=1000,
				 train_interval=1, memory_interval=1, reward_range=(-np.inf, np.inf),
				 action_range=(-np.inf, np.inf), input_batch_processor=None, enable_terminal=True,
				 tau=.001, random_process=OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=0.3),
				 custom_objects={}):
		if not isinstance(actor, Graph):
			raise ValueError('actor must be a Graph model')
		if not isinstance(critic, Graph):
			raise ValueError('critic must be a Graph model')

		# Detect outputs and input names.
		assert actor.nb_output == 1
		assert critic.nb_output == 1
		self.actor_output_name = actor.outputs.keys()[0]
		self.critic_output_name = critic.outputs.keys()[0]
		shared_input_names = set(actor.inputs.keys()).intersection(set(critic.inputs.keys()))
		assert actor.nb_input == len(shared_input_names)
		assert critic.nb_input == len(shared_input_names) + 1
		self.action_input_name = list(set(critic.inputs.keys()).difference(set(actor.inputs.keys())))[0]

		# Parameters.
		self.n_actions = n_actions
		self.input_batch_processor = input_batch_processor
		self.n_steps = 0  # the number of performed steps
		self.n_steps_critic_warmup = n_steps_critic_warmup
		self.n_steps_actor_warmup = n_steps_actor_warmup
		self.training = False
		self.random_process = random_process
		self.reward_range = reward_range
		self.action_range = action_range
		self.gamma = gamma
		self.tau = tau
		self.batch_size = batch_size
		self.train_interval = train_interval
		self.memory_interval = memory_interval
		self.enable_terminal = enable_terminal

		# Prepare models.
		self.actor = actor
		self.critic = critic
		self.target_actor = clone_model(actor, custom_objects)
		self.target_critic = clone_model(critic, custom_objects)
		
		# Re-compile critic with AdditionalUpdatesOptimizer. This is used to update the target
		# network at the same time the critic is updated.
		optimizer = AdditionalUpdatesOptimizer(critic.optimizer, get_target_model_updates(self.target_critic, self.critic, self.tau))
		self.critic.compile(loss=critic.loss, optimizer=optimizer)

		# Compile actor update function.
		#self.actor_train, self.actor_optimizer = self.compile_actor_update_function(self.actor, self.target_critic)  # TODO: this is not what the paper describes, but this is probably a lot more stable
		self.actor_train, self.actor_optimizer = self.compile_actor_update_function(self.actor, self.critic)
		
		# Book-keeping.
		self.temporal_window = temporal_window
		self.memory_limit = memory_limit
		self.memory = Memory(temporal_window=temporal_window, memory_limit=memory_limit)
		self.reset()

	def compile_actor_update_function(self, actor, critic):
		# Temporarily connect to a large, combined model so that we can compute the gradient and monitor
		# the performance of the actor as evaluated by the critic.
		shared_input_names = set(actor.inputs.keys()).intersection(set(critic.inputs.keys()))
		critic_layer_cache = critic.layer_cache
		actor_layer_cache = actor.layer_cache
		critic.layer_cache = {}
		actor.layer_cache = {}
		for name in shared_input_names:
			critic.inputs[name].previous = actor.inputs[name]
		critic.inputs[self.action_input_name].previous = actor.outputs[self.actor_output_name]
		output = critic.get_output(train=True)[self.critic_output_name]
		if K._BACKEND == 'tensorflow':
			grads = K.gradients(output, actor.trainable_weights)
			grads = [g / float(self.batch_size) for g in grads]
		elif K._BACKEND == 'theano':
			import theano.tensor as T
			grads = T.jacobian(output.flatten(), actor.trainable_weights)
			grads = [K.mean(g, axis=0) for g in grads]
		else:
			raise RuntimeError('unknown backend')
		for name in shared_input_names:
			del critic.inputs[name].previous
		del critic.inputs[self.action_input_name].previous
		critic.layer_cache = critic_layer_cache
		actor.layer_cache = actor_layer_cache
		
		# We now have the gradients (`grads`) of the combined model wrt to the actor's weights and
		# the output (`output`). Compute the necessary updates using a clone of the actor's optimizer.
		optimizer = clone_optimizer(actor.optimizer)
		clipnorm = optimizer.clipnorm if hasattr(optimizer, 'clipnorm') else 0.
		clipvalue = optimizer.clipvalue if hasattr(optimizer, 'clipvalue') else 0.
		
		def get_gradients(loss, params):
			# We want to follow the gradient, but the optimizer goes in the opposite direction to
			# minimize loss. Hence the double inversion.
			assert len(grads) == len(params)
			modified_grads = [-g for g in grads]
			if clipnorm > 0.:
				norm = K.sqrt(sum([K.sum(K.square(g)) for g in modified_grads]))
				modified_grads = [optimizers.clip_norm(g, clipnorm, norm) for g in modified_grads]
			if clipvalue > 0.:
				modified_grads = [K.clip(g, -clipvalue, clipvalue) for g in modified_grads]
			return modified_grads
		
		optimizer.get_gradients = get_gradients
		updates = optimizer.get_updates(actor.trainable_weights, actor.constraints, None)
		updates += get_target_model_updates(self.target_actor, self.actor, self.tau)
		updates += actor.updates  # include other updates of the actor, e.g. for BN

		# Finally, combine it all into a callable function.
		inputs = actor.get_input(train=True)
		if isinstance(inputs, dict):
			inputs = [inputs[name] for name in actor.input_order]
		elif not isinstance(inputs, list):
			inputs = [inputs]
		assert isinstance(inputs, list)
		fn = K.function(inputs, [output], updates=updates)
		return fn, optimizer

	def load_weights(self, filepath):
		filename, extension = os.path.splitext(filepath)
		actor_filepath = filename + '_actor' + extension
		critic_filepath = filename + '_critic' + extension
		self.actor.load_weights(actor_filepath)
		self.critic.load_weights(critic_filepath)
		self.target_actor.set_weights(self.actor.get_weights())
		self.target_critic.set_weights(self.critic.get_weights())

	def save_weights(self, filepath, overwrite=False):
		filename, extension = os.path.splitext(filepath)
		actor_filepath = filename + '_actor' + extension
		critic_filepath = filename + '_critic' + extension
		self.actor.save_weights(actor_filepath, overwrite=overwrite)
		self.critic.save_weights(critic_filepath, overwrite=overwrite)

	# TODO: implement state restoration

	def reset(self):
		self.recent_inputs = deque(maxlen=self.temporal_window)
		self.recent_action = None

	def process_input_batch(self, input_batch):
		if self.input_batch_processor is None:
			return input_batch
		processed_input_batch = self.input_batch_processor(input_batch)
		return processed_input_batch

	def select_action(self, state):
		batch = self.process_input_batch([state])
		action = predict_on_batch(self.actor, batch, output_name=self.actor_output_name).flatten()
		assert action.shape == (self.n_actions,)

		# Apply noise, if a random process is set.
		if self.training and self.random_process is not None:
			noise = self.random_process.sample()
			assert noise.shape == action.shape
			action += noise

		return np.clip(action, self.action_range[0], self.action_range[1])

	def forward(self, current_input):
		# Select an action.
		while len(self.recent_inputs) < self.recent_inputs.maxlen:
			# Not enough data, fill the recent_inputs array with
			# the current input. This allows us to immediately
			# perform a policy action instead of falling back
			# to random actions.
			self.recent_inputs.append(deepcopy(current_input))
		state = list(self.recent_inputs)[1:] + [current_input]
		assert len(state) == self.temporal_window
		action = self.select_action(state)
		
		# Book-keeping.
		self.recent_inputs.append(current_input)
		self.recent_action = action
		
		return action

	def backward(self, reward, terminal=False):
		self.n_steps += 1
		stats = [0., 0.]
		if not self.training:
			# We're done here
			return stats

		# Clip the reward to be in reward_range.
		reward = min(max(reward, self.reward_range[0]), self.reward_range[1])

		# Store most recent experience in memory.
		if self.n_steps % self.memory_interval == 0:
			self.memory.append(self.recent_inputs[-1], self.recent_action, reward, terminal)
		
		# Train the network on a single stochastic batch.
		can_train_either = self.n_steps > self.n_steps_actor_warmup or self.n_steps > self.n_steps_critic_warmup
		if can_train_either and self.n_steps % self.train_interval == 0:
			experiences = self.memory.sample(self.batch_size)
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
			state0_batch = self.process_input_batch(state0_batch)
			state1_batch = self.process_input_batch(state1_batch)
			terminal_batch = np.array(terminal_batch)
			reward_batch = np.array(reward_batch)
			action_batch = np.array(action_batch)
			assert reward_batch.shape == (self.batch_size,)
			assert terminal_batch.shape == reward_batch.shape
			assert action_batch.shape == (self.batch_size, self.n_actions)

			# Update critic, if warm up is over.
			if self.n_steps > self.n_steps_critic_warmup:
				# Predict actions.
				target_actions = predict_on_batch(self.target_actor, state1_batch, output_name=self.actor_output_name)
				target_actions = np.clip(target_actions, self.action_range[0], self.action_range[1])
				assert target_actions.shape == (self.batch_size, self.n_actions)
				
				# Predict Q values.
				state1_batch_with_action = dict(state1_batch)  # shallow copy
				state1_batch_with_action[self.action_input_name] = target_actions
				target_q_values = predict_on_batch(self.target_critic, state1_batch_with_action, output_name=self.critic_output_name).flatten()
				assert target_q_values.shape == (self.batch_size,)
				
				# Compute r_t + gamma * max_a Q(s_t+1, a) and update the target ys accordingly,
				# but only for the affected output units (as given by action_batch).
				discounted_reward_batch = self.gamma * target_q_values
				if self.enable_terminal:
					# Set discounted reward to zero for all states that were terminal.
					discounted_reward_batch *= terminal_batch
				assert discounted_reward_batch.shape == reward_batch.shape
				targets = (reward_batch + discounted_reward_batch).reshape(self.batch_size, 1)
				
				# Perform a single batch update on the critic network.
				state0_batch_with_action = dict(state0_batch)  # shallow copy
				state0_batch_with_action[self.action_input_name] = action_batch
				stats[0] = train_on_batch(self.critic, state0_batch_with_action, targets, output_name=self.critic_output_name)

			# Update actor, if warm up is over.
			if self.n_steps > self.n_steps_actor_warmup:
				q_values = self.actor_train([state0_batch[name] for name in self.actor.input_order])[0].flatten()
				assert q_values.shape == (self.batch_size,)
				stats[1] = np.mean(q_values)

		return stats


def a3c_select_action(actor, processed_input, action_dim, action_range):
	batch = processed_input
	outputs = actor.predict_on_batch(batch)
	means = outputs['output_means'].flatten()
	variances = outputs['output_variances'].flatten()
	assert means.shape == (action_dim,)
	assert variances.shape == (action_dim,) or variances.shape == (1,)
	action = np.random.normal(means, np.sqrt(variances) + 1e-8, size=(action_dim,))
	assert action.shape == (action_dim,)
	return np.clip(action, action_range[0], action_range[1])


class A3CAgent(Agent):
	def __init__(self, action_dim, actor, critic, gamma=.9, beta=1e-4, tau=1., reward_range=(-np.inf, np.inf),
				 action_range=(-np.inf, np.inf),
				 input_batch_processor=None, custom_objects={}, batch_size=5, nb_threads=4, queue_maxsize=5):
		self.action_dim = action_dim
		self.critic = critic
		self.actor = actor
		
		self.gamma = gamma
		self.beta = beta
		self.tau = tau
		self.reward_range = reward_range
		self.action_range = action_range
		self.input_batch_processor = input_batch_processor
		self.custom_objects = custom_objects
		self.batch_size = batch_size
		self.nb_threads = nb_threads
		self.queue_maxsize = queue_maxsize

	def fit(self, game, nb_epoch=100, action_repetition=1, callbacks=[], verbose=1):
		if action_repetition < 1:
			raise ValueError('action_repetition must be >= 1')

		# Create copies of the game.
		games = [copy(game) for _ in xrange(self.nb_threads - 1)]
		games.append(game)
		assert len(games) == self.nb_threads

		# Create queue and internal workers that are used for the actual training.
		queue = Queue(maxsize=self.queue_maxsize)
		workers = []
		thread_callbacks = []
		for worker_id in xrange(self.nb_threads):
			worker = _A3CWorker(worker_id, self.action_dim, self.actor, self.critic, queue, gamma=self.gamma, beta=self.beta,
				tau=self.tau, reward_range=self.reward_range, action_range=self.action_range, input_batch_processor=self.input_batch_processor,
				custom_objects=self.custom_objects, batch_size=self.batch_size)
			workers.append(worker)

		# Create updates
		# TODO: move this to the compile step
		import tensorflow as tf
		actor_gradients = [K.placeholder(shape=K.int_shape(p)) for p in self.actor.trainable_weights]
		actor_gradients_names = [v.name for v in actor_gradients]
		actor_optimizer = self.actor.optimizer
		actor_optimizer.get_gradients = lambda loss, params: actor_gradients
		updates = actor_optimizer.get_updates(self.actor.trainable_weights, self.actor.constraints, None)
		# Do not include the updates as defined by the critic here, since we have pre-computed gradients
		# and the updates would probably be wrong (e.g. for batch normalization).
		actor_updates = [tf.assign(a, b) for a, b in updates]
		critic_gradients = [K.placeholder(shape=K.int_shape(p)) for p in self.critic.trainable_weights]
		critic_gradients_names = [v.name for v in critic_gradients]
		critic_optimizer = self.critic.optimizer
		critic_optimizer.get_gradients = lambda loss, params: critic_gradients
		updates = critic_optimizer.get_updates(self.critic.trainable_weights, self.critic.constraints, None)
		# Do not include the updates as defined by the critic here, since we have pre-computed gradients
		# and the updates would probably be wrong (e.g. for batch normalization).
		critic_updates = [tf.assign(a, b) for a, b in updates]
		combined_updates = actor_updates + critic_updates
		combined_names = actor_gradients_names + critic_gradients_names

		# Prepare callbacks.
		if verbose > 2:
			callbacks += [TrainEpochLogger()]
		elif verbose == 1:
			callbacks += [TrainIntervalLogger()]
		callbacks = cbks.CallbackList(callbacks)
		callbacks._set_model(self)
		callbacks._set_params({
			'nb_epoch': nb_epoch,
		})
		callbacks.on_train_begin()

		# Spawn a thread per worker.
		threads = []
		for worker, game in zip(workers, games):
			t = threading.Thread(target=worker.fit, args=(game, action_repetition))
			t.daemon = True
			threads.append(t)
			t.start()

		# Process updates delivered from the worker threads.
		global_epoch_idx = 0
		global_batch_idx = 0
		worker_epoch_indexes = {}
		worker_epoch_rewards = {}
		while global_epoch_idx < nb_epoch:
			# Get update package from one of the threads and run the update.
			data = queue.get()

			# Get the current epoch.
			worker_id = data['worker_id']
			if worker_id not in worker_epoch_indexes:
				worker_epoch_indexes[worker_id] = global_epoch_idx
				worker_epoch_rewards[worker_id] = 0.
				callbacks.on_epoch_begin(global_epoch_idx)
				global_epoch_idx += 1
			current_epoch_idx = worker_epoch_indexes[worker_id]

			# Process each "batch".
			assert len(data['inputs']) == self.batch_size
			for idx, (i, a, r, t) in enumerate(zip(data['inputs'], data['actions'], data['rewards'], data['terminals'])):
				callbacks.on_batch_begin(global_batch_idx)
				stats = 0. if idx < self.batch_size - 1 else data['critic_loss']
				batch_logs = {
					'action': a,
					'input': i,
					'reward': r,
					'stats': stats,
					'epoch': current_epoch_idx,
				}
				worker_epoch_rewards[worker_id] += r
				callbacks.on_batch_end(global_batch_idx, batch_logs)
				global_batch_idx += 1

				# If the state is terminal, finalize the current epoch and start a new one.
				if t:
					epoch_logs = {
						'total_reward': worker_epoch_rewards[worker_id],
					}
					callbacks.on_epoch_end(current_epoch_idx, epoch_logs)
					global_epoch_idx += 1
					worker_epoch_indexes[worker_id] = global_epoch_idx
					worker_epoch_rewards[worker_id] = 0.
					current_epoch_idx = global_epoch_idx
					callbacks.on_epoch_begin(current_epoch_idx)

			# Perform the actual updates.
			combined_inputs = data['actor_grads'] + data['critic_grads']
			feed_dict = dict(zip(combined_names, combined_inputs))
			K.get_session().run(combined_updates, feed_dict=feed_dict)		

		for worker in workers:
			worker.running = False
		callbacks.on_train_end()

	def reset(self):
		#self.actor.reset_states()
		pass

	def process_input_batch(self, input_batch):
		if self.input_batch_processor is None:
			return input_batch
		processed_input_batch = self.input_batch_processor(input_batch)
		return processed_input_batch

	def forward(self, current_input):
		if self.training:
			raise RuntimeError('Use the `fit` method of the A3CAgent class to perform training.')
		batch = self.process_input_batch([current_input])
		return a3c_select_action(self.actor, batch, self.action_dim, self.action_range)

	def backward(self, reward, terminal=False):
		if self.training:
			raise RuntimeError('Use the `fit` method of the A3CAgent class to perform training.')
		# Training is done by the workers.
		pass

	def save_weights(self, filepath, overwrite=False):
		filename, extension = os.path.splitext(filepath)
		actor_filepath = filename + '_actor' + extension
		critic_filepath = filename + '_critic' + extension
		self.actor.save_weights(actor_filepath, overwrite=overwrite)
		self.critic.save_weights(critic_filepath, overwrite=overwrite)

	def load_weights(self, filepath, overwrite=False):
		filename, extension = os.path.splitext(filepath)
		actor_filepath = filename + '_actor' + extension
		critic_filepath = filename + '_critic' + extension
		self.actor.load_weights(actor_filepath)
		self.critic.load_weights(critic_filepath)


class _A3CWorker(object):
	def __init__(self, id, action_dim, global_actor, global_critic, update_queue, gamma=.9, beta=1e-4, tau=1.,
				 reward_range=(-np.inf, np.inf), action_range=(-np.inf, np.inf), input_batch_processor=None, enable_terminal=True,
				 custom_objects={}, batch_size=5, enable_bootstrapping=False):
		self.id = id
		self.global_actor = global_actor
		self.global_critic = global_critic
		self.update_queue = update_queue
		self.local_actor = clone_model(global_actor, custom_objects)
		self.local_critic = clone_model(global_critic, custom_objects)
		self.n_steps = 0

		self.enable_bootstrapping = enable_bootstrapping
		self.action_dim = action_dim
		self.tau = tau
		self.beta = beta
		self.gamma = gamma
		self.reward_range = reward_range
		self.action_range = action_range
		self.input_batch_processor = input_batch_processor
		self.enable_terminal = enable_terminal
		self.batch_size = batch_size

		self.reward_accumulator = []
		self.input_accumulator = []
		self.action_accumulator = []
		self.terminal_accumulator = []

		self.actor_gradient_func = _A3CWorker.compile_actor_gradient_func(self.local_actor, action_dim, beta,
			clipnorm=getattr(global_actor, 'clipnorm', 0.), clipvalue=getattr(global_actor, 'clipvalue', 0.))
		self.critic_gradient_func, self.critic_gradient_tensors = _A3CWorker.compile_critic_gradient_func(self.local_critic,
			clipnorm=getattr(global_critic, 'clipnorm', 0.), clipvalue=getattr(global_critic, 'clipvalue', 0.))

		if K._BACKEND == 'tensorflow':
			import tensorflow as tf
			updates = get_target_model_updates(target=self.local_actor, source=self.global_actor, tau=tau)
			updates += get_target_model_updates(target=self.local_critic, source=self.global_critic, tau=tau)
			self.weight_updates = []
			for a, b in updates:
				self.weight_updates.append(tf.assign(a, b))
		self.update_local_weights()

	def save_weights(self, filepath, overwrite=False):
		filename, extension = os.path.splitext(filepath)
		actor_filepath = filename + '_actor' + extension
		critic_filepath = filename + '_critic' + extension
		self.local_actor.save_weights(actor_filepath, overwrite=overwrite)
		self.local_critic.save_weights(critic_filepath, overwrite=overwrite)

	def update_local_weights(self):
		# TODO: do we need locking?
		if K._BACKEND == 'theano':
			self.local_actor.set_weights(self.global_actor.get_weights())
			self.local_critic.set_weights(self.global_critic.get_weights())
		elif K._BACKEND == 'tensorflow':
			K.get_session().run(self.weight_updates)

	@staticmethod
	def compile_critic_gradient_func(critic, clipnorm=0., clipvalue=0.):
		# TODO: this is just a very basic implementation. Does not implement regularization and all that right now.
		output = critic.get_output(train=True)
		if type(output) == dict:
			output = output[output.keys()[0]]
		ys = K.placeholder(shape=(None,))
		loss = K.mean(objectives.get('mse')(output, ys))
		grads = K.gradients(loss, critic.trainable_weights)

		# We now have the gradients (`grads`) of the combined model wrt to the actor's weights.
		# Apply gradient constraints.
		if clipnorm > 0.:
			norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
			grads = [optimizers.clip_norm(g, clipnorm, norm) for g in grads]
		if clipvalue > 0.:
			grads = [K.clip(g, -clipvalue, clipvalue) for g in grads]

		# Since we want to get the loss as well as the gradients, we use `updates` to write out the
		# gradient into a target tensor.
		grad_tensors = [K.zeros(shape=K.int_shape(w)) for w in critic.trainable_weights]
		updates = [(t, g) for t, g in zip(grad_tensors, grads)]
		updates += critic.state_updates

		# Finally, combine it all into a callable function.
		inputs = critic.get_input(train=True)
		if isinstance(inputs, dict):
			inputs = [inputs[name] for name in critic.input_order]
		elif not isinstance(inputs, list):
			inputs = [inputs]
		assert isinstance(inputs, list)
		fn = K.function(inputs + [ys], [loss, output], updates=updates)
		return fn, grad_tensors

	@staticmethod
	def compile_actor_gradient_func(actor, action_dim, beta, clipnorm=0., clipvalue=0.):
		EPS = 1e-10  # TODO: what is a good value for this?

		Vs = K.placeholder(shape=(1,))
		Rs = K.placeholder(shape=(1,))
		actions = K.placeholder(shape=(1, action_dim))
		outputs = actor.get_output(train=True)
		means = outputs['output_means']
		variances = outputs['output_variances'] + EPS
		
		# Compute the probability of the action under a normal distribution.
		output_proba = 1. / K.sqrt(2. * np.pi * variances) * K.exp(-K.square(actions - means) / (2. * variances))
		output_log_proba = K.log(output_proba + EPS)

		if K._BACKEND == 'tensorflow':
			grads = [g * (Rs - Vs) for g in K.gradients(output_log_proba, actor.trainable_weights)]
		elif K._BACKEND == 'theano':
			# TODO: implement working theano
			# TODO: adapt this work with many outputs
			raise NotImplementedError()
			# import theano.tensor as T
			# grads = T.jacobian(output_proba.flatten(), actor.trainable_weights)
			# grads = [K.mean(g, axis=0) for g in grads]
			# grads = [g * (Rs - Vs) for g in grads]
			# TODO: Old Theano-only code that was able to handle batches of size >= 1 but didn't work with LSTMs.
			# raw_grads = T.jacobian(K.log(output_proba.flatten()), actor.trainable_weights)
			# grads = []
			# for weight_idx in xrange(len(actor.trainable_weights)):
			# 	grad = None
			# 	for sample_idx in xrange(batch_size):
			# 		start_idx = sample_idx * action_dim
			# 		# TODO: it's not really clear to me how to flatten this properly. sum or mean?!
			# 		g = K.mean(raw_grads[weight_idx][start_idx:start_idx + action_dim], axis=0)
			# 		g *= (Rs[sample_idx] - Vs[sample_idx])
			# 		grad = grad + g if grad is not None else g
			# 	grad /= float(batch_size)
			# 	grads.append(grad)

		# Compute gradient for the regularization term.
		regularizer = -.5 * (K.log(2. * np.pi * variances) + 1) + 0. * means
		if K._BACKEND == 'tensorflow':
			regularizer_grads = K.gradients(regularizer, actor.trainable_weights)
		elif K._BACKEND == 'theano':
			# TODO: implement working theano
			raise NotImplementedError()
			#import theano.tensor as T
			#regularizer_grads = T.jacobian(regularizer.flatten(), actor.trainable_weights)
		
		# Combine grads.
		grads = [g + beta * rg for g, rg in zip(grads, regularizer_grads)]
		grads = [-g for g in grads]  # we want to follow the gradient (double inversion b/c we use optimizer)
		
		# We now have the gradients (`grads`) of the combined model wrt to the actor's weights.
		# Apply gradient constraints.
		if clipnorm > 0.:
			norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
			grads = [optimizers.clip_norm(g, clipnorm, norm) for g in grads]
		if clipvalue > 0.:
			grads = [K.clip(g, -clipvalue, clipvalue) for g in grads]
		
		# Finally, combine it all into a callable function.
		inputs = actor.get_input(train=True)
		if isinstance(inputs, dict):
			inputs = [inputs[name] for name in actor.input_order]
		elif not isinstance(inputs, list):
			inputs = [inputs]
		assert isinstance(inputs, list)
		fn = K.function(inputs + [Vs, Rs, actions], grads)
		return fn

	def select_action(self, current_input):
		batch = self.process_input_batch([current_input])
		return a3c_select_action(self.local_actor, batch, self.action_dim, self.action_range)

	def fit(self, game, action_repetition):
		if action_repetition < 1:
			raise ValueError('action_repetition must be >= 1')

		self.training = True
		game.training = True
		self.running = True
		while self.running:
			ins = game.perceive()
			action = self.forward(ins)
			reward = 0.
			for _ in xrange(action_repetition):
				reward += game.act(action)
				if game.game_over:
					break
			self.backward(reward, terminal=game.game_over)
			if game.game_over:
				game.reset()
				self.reset()

	def reset(self):
		# TODO: reset state of the RNN. The problem here is that using K.set_value() can cause
		# problems with TensorFlow since it changes the Graph.
		#self.local_actor.reset_states()
		#self.local_critic.reset_states()
		pass

	def forward(self, current_input):
		action = self.select_action(current_input)
		self.input_accumulator.append(current_input)
		self.action_accumulator.append(action)
		return action

	def process_input_batch(self, input_batch):
		if self.input_batch_processor is None:
			return input_batch
		processed_input_batch = self.input_batch_processor(input_batch)
		return processed_input_batch

	def backward(self, reward, terminal=False):
		self.n_steps += 1
		self.reward_accumulator.append(reward)
		self.terminal_accumulator.append(terminal)
		assert len(self.reward_accumulator) == len(self.input_accumulator)
		assert len(self.reward_accumulator) == len(self.terminal_accumulator)
		assert len(self.reward_accumulator) == len(self.action_accumulator)
		
		perform_update = self.training and len(self.input_accumulator) > self.batch_size
		if not perform_update:
			return 0.

		# We have one more data point to bootstrap from.
		assert len(self.input_accumulator) == self.batch_size + 1

		# Accumulate data for gradient computation.
		inputs = self.process_input_batch(self.input_accumulator)
		#Vs = predict_on_batch(self.local_critic, inputs).flatten().tolist()
		#if self.enable_bootstrapping:
		#	R = 0. if self.terminal_accumulator[-1] else Vs[-1]
		#else:
		#	R = 0.
		R = 0.
		Rs = [R]
		for r, t in zip(reversed(self.reward_accumulator[:-1]), reversed(self.terminal_accumulator[:-1])):
			R = r + self.gamma * R if not t else r
			Rs.append(R)
		Rs = list(reversed(Rs))

		# Remove latest value, which we have no use for.
		inputs = np.array(self.input_accumulator[:-1])
		processed_inputs = self.process_input_batch(inputs)
		actions = np.array(self.action_accumulator[:-1])
		rewards = np.array(self.reward_accumulator[:-1])
		terminals = np.array(self.terminal_accumulator[:-1])
		Rs = np.array(Rs[:-1])

		# Ensure that everything is fine and enqueue for update.
		for v in processed_inputs.values():
			assert v.shape[0] == self.batch_size
		assert Rs.shape == (self.batch_size,)
		assert rewards.shape == (self.batch_size,)
		assert actions.shape == (self.batch_size, self.action_dim)
		assert terminals.shape == (self.batch_size,)

		# Compute critic gradients and Vs.
		critic_ins = [processed_inputs[name] for name in self.local_critic.input_order]
		critic_loss, Vs = self.critic_gradient_func(critic_ins + [Rs])
		Vs = Vs.flatten()
		assert Vs.shape == (self.batch_size,)
		critic_grads = [K.get_value(t) for t in self.critic_gradient_tensors]

		# Compute actor gradients. We do this sample by sample b/c the actor could potentially use
		# a stateful RNN.
		actor_grads = None
		for idx in xrange(self.batch_size):
			actor_ins = [processed_inputs[name][idx:idx+1] for name in self.local_actor.input_order]
			grads = self.actor_gradient_func(actor_ins + [Vs[idx:idx+1], Rs[idx:idx+1], actions[idx:idx+1]])
			if actor_grads is None:
				actor_grads = grads
			else:
				actor_grads = [ag + g for ag, g in zip(actor_grads, grads)]
		actor_grads = [g / float(self.batch_size) for g in actor_grads]  # TODO: do we need to divide?
		# TODO: gradient clipping should be performed here, since adding the gradients can change things
		
		# Send gradients off for asynchronous weight updates. Also include actions and rewards for stats.
		data = {
			'worker_id': self.id,
			'actor_grads': actor_grads,
			'critic_grads': critic_grads,
			'critic_loss': critic_loss,
			'inputs': inputs,
			'actions': actions,
			'rewards': rewards,
			'terminals': terminals,
		}
		self.update_queue.put(data)
		
		# Reset state for next update round. We keep the latest data point around since we haven't
		# used it.
		self.input_accumulator = [self.input_accumulator[-1]]
		self.action_accumulator = [self.action_accumulator[-1]]
		self.terminal_accumulator = [self.terminal_accumulator[-1]]
		self.reward_accumulator = [self.reward_accumulator[-1]]
		
		self.update_local_weights()
		return critic_loss
