import warnings

import timeit
import numpy as np

from keras.callbacks import Callback as KerasCallback, CallbackList as KerasCallbackList


class Callback(KerasCallback):
	def on_episode_begin(self, episode, logs={}):
		pass

	def on_episode_end(self, episode, logs={}):
		pass

	def on_step_begin(self, step, logs={}):
		pass

	def on_step_end(self, step, logs={}):
		pass


class CallbackList(KerasCallbackList):
	def on_episode_begin(self, episode, logs={}):
		for callback in self.callbacks:
			# Check if callback supports the more appropriate `on_episode_begin` callback.
			# If not, fall back to `on_epoch_begin` to be compatible with built-in Keras callbacks.
			if callable(getattr(callback, 'on_episode_begin', None)):
				callback.on_episode_begin(episode, logs=logs)
			else:
				callback.on_epoch_begin(episode, logs=logs)

	def on_episode_end(self, episode, logs={}):
		for callback in self.callbacks:
			# Check if callback supports the more appropriate `on_episode_end` callback.
			# If not, fall back to `on_epoch_end` to be compatible with built-in Keras callbacks.
			if callable(getattr(callback, 'on_episode_end', None)):
				callback.on_episode_end(episode, logs=logs)
			else:
				callback.on_epoch_end(episode, logs=logs)

	def on_step_begin(self, step, logs={}):
		for callback in self.callbacks:
			# Check if callback supports the more appropriate `on_step_begin` callback.
			# If not, fall back to `on_batch_begin` to be compatible with built-in Keras callbacks.
			if callable(getattr(callback, 'on_step_begin', None)):
				callback.on_step_begin(step, logs=logs)
			else:
				callback.on_batch_begin(step, logs=logs)

	def on_step_end(self, step, logs={}):
		for callback in self.callbacks:
			# Check if callback supports the more appropriate `on_step_end` callback.
			# If not, fall back to `on_batch_end` to be compatible with built-in Keras callbacks.
			if callable(getattr(callback, 'on_step_end', None)):
				callback.on_step_end(step, logs=logs)
			else:
				callback.on_batch_end(step, logs=logs)


class TestLogger(Callback):
	def on_episode_end(self, episode, logs):
		template = 'Episode {0}: reward: {1:.3f}, steps: {2}'
		variables = [
			episode + 1,
			logs['total_reward'],
			logs['nb_steps'],
		]
		print(template.format(*variables))


class TrainEpisodeLogger(Callback):
	def __init__(self):
		# Some algorithms compute multiple episodes at once since they are multi-threaded.
		# We therefore use a dictionary that is indexed by the episode to separate episodes
		# from each other.
		self.episode_start = {}
		self.observations = {}
		self.rewards = {}
		self.actions = {}
		self.metrics = {}

	def on_train_begin(self, logs):
		self.train_start = timeit.default_timer()
		self.metrics_names = self.model.metrics_names
		print('Training for {} episodes ...'.format(self.params['nb_episodes']))

	def on_train_end(self, logs):
		duration = timeit.default_timer() - self.train_start
		print('done, took {0:.3f} seconds'.format(duration))

	def on_episode_begin(self, episode, logs):
		self.episode_start[episode] = timeit.default_timer()
		self.observations[episode] = []
		self.rewards[episode] = []
		self.actions[episode] = []
		self.metrics[episode] = []

	def on_episode_end(self, episode, logs):
		duration = timeit.default_timer() - self.episode_start[episode]
		steps = len(self.observations[episode])

		# Format all metrics.
		metrics = np.array(self.metrics[episode])
		metrics_template = ''
		metrics_variables = []
		with warnings.catch_warnings():
			warnings.filterwarnings('error')
			for idx, name in enumerate(self.metrics_names):
				if idx > 0:
					metrics_template += ', '
				try:
					value = np.nanmean(metrics[:, idx])
					metrics_template += '{}: {:f}'
				except Warning:
					value = '--'
					metrics_template += '{}: {}'
				metrics_variables += [name, value]
					
		metrics_text = metrics_template.format(*metrics_variables)
		
		template = 'Episode {episode}: {duration:.3f}s, steps: {steps}, steps per second: {sps:.0f}, total reward: {total_reward:.3f}, reward: {reward:.3f} [{reward_min:.3f}, {reward_max:.3f}], action: {action:.3f} [{action_min:.3f}, {action_max:.3f}], observations: {obs:.3f} [{obs_min:.3f}, {obs_max:.3f}], {metrics}'
		variables = {
			'episode': episode + 1,
			'duration': duration,
			'steps': steps,
			'sps': float(steps) / duration,
			'total_reward': np.sum(self.rewards[episode]),
			'reward': np.mean(self.rewards[episode]),
			'reward_min': np.min(self.rewards[episode]),
			'reward_max': np.max(self.rewards[episode]),
			'action': np.mean(self.actions[episode]),
			'action_min': np.min(self.actions[episode]),
			'action_max': np.max(self.actions[episode]),
			'obs': np.mean(self.observations[episode]),
			'obs_min': np.min(self.observations[episode]),
			'obs_max': np.max(self.observations[episode]),
			'metrics': metrics_text,
		}
		print(template.format(**variables))

		# Free up resources.
		del self.episode_start[episode]
		del self.observations[episode]
		del self.rewards[episode]
		del self.actions[episode]
		del self.metrics[episode]

	def on_step_end(self, step, logs):
		episode = logs['episode']
		self.observations[episode].append(np.concatenate(logs['observation']))
		self.rewards[episode].append(logs['reward'])
		self.actions[episode].append(logs['action'])
		self.metrics[episode].append(logs['metrics'])

# class TrainIntervalLogger(Callback):
# 	def __init__(self, interval=10000):
# 		self.interval = interval
# 		self.steps = 0
# 		self.reset()

# 	def reset(self):
# 		self.interval_start = timeit.default_timer()
# 		self.observations = []
# 		self.rewards = []
# 		self.actions = []
# 		self.metrics = []
# 		self.average_qs = []

# 	def on_train_begin(self, logs):
# 		self.train_start = timeit.default_timer()
# 		print('Training for {} episodes ...'.format(self.params['nb_episode']))

# 	def on_train_end(self, logs):
# 		duration = timeit.default_timer() - self.train_start
# 		print('done, took {0:.3f} seconds'.format(duration))

# 	def print_report(self):
# 		if len(self.average_qs) == 0:
# 			self.average_qs.append(0.)
# 		if len(self.metrics) == 0:
# 			self.metrics.append(0.)
# 		duration = timeit.default_timer() - self.interval_start
# 		interval_steps = len(self.observations)
# 		template = '{0}: {1:.3f}s, interval steps: {2}, steps per second: {3:.0f}, reward: {4:.3f} [{5:.3f}, {6:.3f}], loss: {7:.3f}, avg q: {14:.3f} action: {8:.3f} [{9:.3f}, {10:.3f}], observations: {11:.3f} [{12:.3f}, {13:.3f}]'
# 		variables = [
# 			self.steps,
# 			duration,
# 			interval_steps,
# 			float(interval_steps) / duration,
# 			np.sum(self.rewards),
# 			np.min(self.rewards),
# 			np.max(self.rewards),
# 			np.sum(self.metrics),
# 			np.mean(self.actions),
# 			np.min(self.actions),
# 			np.max(self.actions),
# 			np.mean(self.observations),
# 			np.min(self.observations),
# 			np.max(self.observations),
# 			np.mean(self.average_qs),
# 		]
# 		print(template.format(*variables))

# 	def on_step_end(self, step, logs):
# 		if self.steps % self.interval == 0:
# 			if self.steps > 0:
# 				self.print_report()
# 			self.reset()

# 		# Book-keeping.
# 		self.steps += 1
# 		self.observations.append(np.concatenate(logs['observation']))
# 		self.rewards.append(logs['reward'])
# 		self.actions.append(logs['action'])
# 		stats = logs['stats']
# 		if not isinstance(stats, (list, tuple)):
# 			stats = [stats]
# 		self.metrics.append(stats[0])
# 		if len(stats) >= 2:
# 			self.average_qs.append(stats[1])


class Visualizer(Callback):
	def on_step_end(self, step, logs):
		self.params['env'].render(mode='human')
