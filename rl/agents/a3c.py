import keras.backend as K

from rl.core import Agent


class DiscreteA3CAgent(Agent):
	pass

class ContinuousA3CAgent(Agent):
	def __init__(self, nb_actions, actor, critic, gamma=.9, beta=1e-4,
				 target_model_update=1., reward_range=(-np.inf, np.inf), processor=None,
				 custom_objects={}, batch_size=5, enable_bootstrapping=False):
		super(A3CAgent, self).__init__()

		self.actor = actor
		self.critic = critic

		# Parameters.
		self.nb_actions = nb_actions
		self.enable_bootstrapping = enable_bootstrapping
		self.target_model_update = target_model_update
		self.beta = beta
		self.gamma = gamma
		self.reward_range = reward_range
		self.processor = processor
		self.batch_size = batch_size

		self.reward_accumulator = []
		self.observation_accumulator = []
		self.action_accumulator = []
		self.terminal_accumulator = []

		# self.actor_gradient_func = _A3CWorker.compile_actor_gradient_func(self.local_actor, action_dim, beta,
		# 	clipnorm=getattr(global_actor, 'clipnorm', 0.), clipvalue=getattr(global_actor, 'clipvalue', 0.))
		# self.critic_gradient_func, self.critic_gradient_tensors = _A3CWorker.compile_critic_gradient_func(self.local_critic,
		# 	clipnorm=getattr(global_critic, 'clipnorm', 0.), clipvalue=getattr(global_critic, 'clipvalue', 0.))

		# if K._BACKEND == 'tensorflow':
		# 	import tensorflow as tf
		# 	updates = get_target_model_updates(target=self.local_actor, source=self.global_actor, tau=tau)
		# 	updates += get_target_model_updates(target=self.local_critic, source=self.global_critic, tau=tau)
		# 	self.weight_updates = []
		# 	for a, b in updates:
		# 		self.weight_updates.append(tf.assign(a, b))
		# self.update_local_weights()

	def compile(self, optimizer, metrics=[]):
		self.local_actor = clone_model(actor, custom_objects)
		self.local_actor.compile(optimizer='sgd', loss='mse')  # never used for optimization
		self.local_critic = clone_model(critic, custom_objects)
		self.local_critic.compile(optimizer='sgd', loss='mse')  # never used for optimization

		#TODO: check if critic and actor are already compiled
		self.critic.compile(optimizer=optimizer, loss='mse', metrics=metrics)

		Vs = K.placeholder(shape=(1,))
		Rs = K.placeholder(shape=(1,))
		actions = K.placeholder(shape=(1, action_dim))
		outputs = actor.get_output(train=True)
		means = outputs['output_means']
		variances = outputs['output_variances'] + K.epsilon()
		
		# Compute the probability of the action under a normal distribution.
		pdf = 1. / K.sqrt(2. * np.pi * variances) * K.exp(-K.square(actions - means) / (2. * variances))
		log_pdf = K.log(pdf + K.epsilon())

		if K._BACKEND == 'tensorflow':
			grads = [g * (Rs - Vs) for g in K.gradients(log_pdf, self.actor.trainable_weights)]
		elif K._BACKEND == 'theano':
			raise NotImplementedError()
		else:
			raise RuntimeError('Unknown Keras backend "{}".'.format(K._BACKEND))

		# Compute gradient for the regularization term. We have to add the means here since otherwise
		# computing the gradient fails due to an unconnected graph.
		regularizer = -.5 * (K.log(2. * np.pi * variances) + 1) + 0. * means
		if K._BACKEND == 'tensorflow':
			regularizer_grads = K.gradients(regularizer, self.actor.trainable_weights)
		elif K._BACKEND == 'theano':
			raise NotImplementedError()
		else:
			raise RuntimeError('Unknown Keras backend "{}".'.format(K._BACKEND))
		
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


		self.compiled = True

	def save_weights(self, filepath, overwrite=False):
		filename, extension = os.path.splitext(filepath)
		actor_filepath = filename + '_actor' + extension
		critic_filepath = filename + '_critic' + extension
		self.local_actor.save_weights(actor_filepath, overwrite=overwrite)
		self.local_critic.save_weights(critic_filepath, overwrite=overwrite)

	def load_weights(self, filepath):
		filename, extension = os.path.splitext(filepath)
		actor_filepath = filename + '_actor' + extension
		critic_filepath = filename + '_critic' + extension
		self.local_actor.load_weights(actor_filepath)
		self.local_critic.load_weights(critic_filepath)
		# TODO: update global weights?!

	def update_local_weights(self):
		# TODO: update this!
		# TODO: do we need locking?
		if K._BACKEND == 'theano':
			self.local_actor.set_weights(self.global_actor.get_weights())
			self.local_critic.set_weights(self.global_critic.get_weights())
		elif K._BACKEND == 'tensorflow':
			K.get_session().run(self.weight_updates)

	def select_action(self, current_input):
		# TODO: policy?
		batch = self.process_input_batch([current_input])
		return a3c_select_action(self.local_actor, batch, self.action_dim, self.action_range)

	def reset_states(self):
		self.local_actor.reset_states()
		self.local_critic.reset_states()

	def forward(self, observation):
		action = self.select_action(observation)
		self.observation_accumulator.append(observation)
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
		assert len(self.reward_accumulator) == len(self.observation_accumulator)
		assert len(self.reward_accumulator) == len(self.terminal_accumulator)
		assert len(self.reward_accumulator) == len(self.action_accumulator)
		
		perform_update = self.training and len(self.observation_accumulator) > self.batch_size
		if not perform_update:
			return 0.

		# We have one more data point to bootstrap from.
		assert len(self.observation_accumulator) == self.batch_size + 1

		# Accumulate data for gradient computation.
		inputs = self.process_input_batch(self.observation_accumulator)
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
		inputs = np.array(self.observation_accumulator[:-1])
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
		self.observation_accumulator = [self.observation_accumulator[-1]]
		self.action_accumulator = [self.action_accumulator[-1]]
		self.terminal_accumulator = [self.terminal_accumulator[-1]]
		self.reward_accumulator = [self.reward_accumulator[-1]]
		
		self.update_local_weights()
		return critic_loss