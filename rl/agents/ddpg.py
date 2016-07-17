from collections import deque
import os

import numpy as np
import keras.backend as K

from rl.core import Agent
from rl.random import OrnsteinUhlenbeckProcess
from rl.util import clone_model, clone_optimizer, AdditionalUpdatesOptimizer


def mean_q(y_true, y_pred):
	return K.mean(K.max(y_pred, axis=-1))


def get_target_model_updates(target, source, tau):
	target_weights = target.trainable_weights + sum([l.non_trainable_weights for l in target.layers], [])
	source_weights = source.trainable_weights + sum([l.non_trainable_weights for l in source.layers], [])
	assert len(target_weights) == len(source_weights)

	# Create updates.
	updates = []
	for tw, sw in zip(target_weights, source_weights):
		updates.append((tw, tau * sw + (1. - tau) * tw))
	return updates


# Deep DPG as described by Lillicrap et al. (2015)
# http://arxiv.org/pdf/1509.02971v2.pdf
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.646.4324&rep=rep1&type=pdf
class DDPGAgent(Agent):
	def __init__(self, nb_actions, actor, critic, critic_action_input, memory, window_length=1,
				 gamma=.99, batch_size=32, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000,
				 train_interval=1, memory_interval=1, reward_range=(-np.inf, np.inf), processor=None,
				 tau=.001, random_process=OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=0.3),
				 custom_model_objects={}):
		if hasattr(actor.output, '__len__') and len(actor.output) > 1:
			raise ValueError('Actor "{}" has more than one output. DDPG expects an actor that has a single output.'.format(actor))
		if hasattr(actor.input, '__len__') and len(actor.input) != 1:
			raise ValueError('Actor "{}" does have too many inputs. The actor must have at exactly one input for the observation.'.format(actor))
		if hasattr(critic.output, '__len__') and len(critic.output) > 1:
			raise ValueError('Critic "{}" has more than one output. DDPG expects a critic that has a single output.'.format(critic))
		if critic_action_input not in critic.input:
			raise ValueError('Critic "{}" does not have designated action input "{}".'.format(critic, critic_action_input))
		if not hasattr(critic.input, '__len__') or len(critic.input) != 2:
			raise ValueError('Critic "{}" does not have enough inputs. The critic must have at exactly two inputs, one for the action and one for the observation.'.format(critic))
		if critic_action_input._keras_shape != actor.output._keras_shape:
			raise ValueError('Critic "{}" and actor "{}" do not have matching shapes')

		super(DDPGAgent, self).__init__()

		# Parameters.
		self.nb_actions = nb_actions
		self.window_length = window_length
		self.processor = processor
		self.nb_steps_warmup_actor = nb_steps_warmup_actor
		self.nb_steps_warmup_critic = nb_steps_warmup_critic
		self.random_process = random_process
		self.reward_range = reward_range
		self.gamma = gamma
		self.tau = tau
		self.batch_size = batch_size
		self.train_interval = train_interval
		self.memory_interval = memory_interval
		self.custom_model_objects = custom_model_objects

		# Related objects.
		self.actor = actor
		self.critic = critic
		self.critic_action_input = critic_action_input
		self.critic_action_input_idx = self.critic.input.index(critic_action_input)
		self.memory = memory
		self.processor = processor

		# State.
		self.compiled = False
		self.reset_states()

	def compile(self, optimizer, metrics=[]):
		metrics += [mean_q]

		if hasattr(optimizer, '__len__'):
			if len(optimizer) != 2:
				raise ValueError('More than two optimizers provided. Please only provide a maximum of two optimizers, the first one for the actor and the second one for the critic.')
			actor_optimizer, critic_optimizer = optimizer
		else:
			actor_optimizer = optimizer
			critic_optimizer = clone_optimizer(optimizer)
		assert actor_optimizer != critic_optimizer

		if len(metrics) == 2 and hasattr(metrics[0], '__len__') and hasattr(metrics[1], '__len__'):
			actor_metrics, critic_metrics = metrics
		else:
			actor_metrics = critic_metrics = metrics

		# Compile target networks. We only use them in feed-forward mode, hence we can pass any
		# optimizer and loss since we never use it anyway.
		self.target_actor = clone_model(self.actor, self.custom_model_objects)
		self.target_actor.compile(optimizer='sgd', loss='mse')
		self.target_critic = clone_model(self.critic, self.custom_model_objects)
		self.target_critic.compile(optimizer='sgd', loss='mse')

		# We also compile the actor. We never optimize the actor using Keras but instead compute
		# the policy gradient ourselves. However, we need the actor in feed-forward mode, hence
		# we also compile it with any optimzer and
		self.actor.compile(optimizer='sgd', loss='mse')

		# Compile the critic. We use the `AdditionalUpdatesOptimizer` to efficiently update the target model.
		critic_updates = get_target_model_updates(self.target_critic, self.critic, self.tau)
		critic_optimizer = AdditionalUpdatesOptimizer(critic_optimizer, critic_updates)
		self.critic.compile(optimizer=critic_optimizer, loss='mse', metrics=critic_metrics)

		# Combine actor and critic so that we can get the policy gradient.
		combined_inputs = []
		critic_inputs = []
		for i in self.critic.input:
			if i == self.critic_action_input:
				combined_inputs.append(self.actor.output)
			else:
				combined_inputs.append(i)
				critic_inputs.append(i)
		combined_output = self.critic(combined_inputs)
		if K._BACKEND == 'tensorflow':
			grads = K.gradients(combined_output, self.actor.trainable_weights)
			grads = [g / float(self.batch_size) for g in grads]  # since TF sums over the batch
		elif K._BACKEND == 'theano':
			import theano.tensor as T
			grads = T.jacobian(combined_output.flatten(), self.actor.trainable_weights)
			grads = [K.mean(g, axis=0) for g in grads]
		else:
			raise RuntimeError('Unknown Keras backend "{}".'.format(K._BACKEND))
		
		# We now have the gradients (`grads`) of the combined model wrt to the actor's weights and
		# the output (`output`). Compute the necessary updates using a clone of the actor's optimizer.
		clipnorm = getattr(actor_optimizer, 'clipnorm', 0.)
		clipvalue = getattr(actor_optimizer, 'clipvalue', 0.)
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
		actor_optimizer.get_gradients = get_gradients
		updates = actor_optimizer.get_updates(self.actor.trainable_weights, self.actor.constraints, None)
		updates += get_target_model_updates(self.target_actor, self.actor, self.tau)
		updates += self.actor.updates  # include other updates of the actor, e.g. for BN

		# Finally, combine it all into a callable function.
		actor_inputs = None
		if not hasattr(self.actor.input, '__len__'):
			actor_inputs = [self.actor.input]
		else:
			actor_inputs = self.actor.input
		self.actor_train_fn = K.function(actor_inputs + critic_inputs, [self.actor.output], updates=updates)
		self.actor_optimizer = actor_optimizer

		self.compiled = True

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

	# TODO: implement pickle

	def reset_states(self):
		self.recent_action = None
		self.recent_observations = deque(maxlen=self.window_length)

	def process_state_batch(self, batch):
		batch = np.array(batch)
		if self.processor is None:
			return batch
		return self.processor.process_state_batch(batch)

	def select_action(self, state):
		batch = self.process_state_batch([state])
		action = self.actor.predict_on_batch(batch).flatten()
		assert action.shape == (self.nb_actions,)

		# Apply noise, if a random process is set.
		if self.training and self.random_process is not None:
			noise = self.random_process.sample()
			assert noise.shape == action.shape
			action += noise

		return action

	def forward(self, observation):
		# TODO: this could be shared with DQN
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
		action = self.select_action(state)  # TODO: move this into policy
		
		# Book-keeping.
		self.recent_observations.append(observation)
		self.recent_action = action
		
		return action

	@property
	def metrics_names(self):
		return self.critic.metrics_names[:]

	def backward(self, reward, terminal=False):
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
		can_train_either = self.step > self.nb_steps_warmup_critic or self.step > self.nb_steps_warmup_actor
		if can_train_either and self.step % self.train_interval == 0:
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
			action_batch = np.array(action_batch)
			assert reward_batch.shape == (self.batch_size,)
			assert terminal_batch.shape == reward_batch.shape
			assert action_batch.shape == (self.batch_size, self.nb_actions)

			# Update critic, if warm up is over.
			if self.step > self.nb_steps_warmup_critic:
				target_actions = self.target_actor.predict_on_batch(state1_batch)
				assert target_actions.shape == (self.batch_size, self.nb_actions)
				state1_batch_with_action = [state1_batch]
				state1_batch_with_action.insert(self.critic_action_input_idx, target_actions)
				#print state1_batch_with_action
				target_q_values = self.target_critic.predict_on_batch(state1_batch_with_action).flatten()
				assert target_q_values.shape == (self.batch_size,)
				
				# Compute r_t + gamma * max_a Q(s_t+1, a) and update the target ys accordingly,
				# but only for the affected output units (as given by action_batch).
				discounted_reward_batch = self.gamma * target_q_values
				discounted_reward_batch *= terminal_batch
				assert discounted_reward_batch.shape == reward_batch.shape
				targets = (reward_batch + discounted_reward_batch).reshape(self.batch_size, 1)
				
				# Perform a single batch update on the critic network.
				state0_batch_with_action = [state0_batch]
				state0_batch_with_action.insert(self.critic_action_input_idx, action_batch)
				metrics = self.critic.train_on_batch(state0_batch_with_action, targets)

			# Update actor, if warm up is over.
			if self.step > self.nb_steps_warmup_actor:
				# TODO: implement metrics for actor
				q_values = self.actor_train_fn([state0_batch, state0_batch])[0].flatten()
				assert q_values.shape == (self.batch_size,)

		return metrics
