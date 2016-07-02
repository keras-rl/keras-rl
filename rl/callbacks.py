import timeit
import numpy as np

from keras.callbacks import Callback


class TestLogger(Callback):
	def on_epoch_end(self, epoch, logs):
		template = 'Epoch {0}: reward: {1:.3f}'
		variables = [
			epoch,
			logs['total_reward'],
		]
		print(template.format(*variables))


class TrainEpochLogger(Callback):
	def __init__(self):
		# Some algorithms compute multiple epochs at once since they are multi-threaded.
		# We therefore use a dictionary that is indexed by the epoch to separate epochs
		# from each other.
		self.epoch_start = {}
		self.inputs = {}
		self.rewards = {}
		self.actions = {}
		self.losses = {}
		self.average_qs = {}

	def on_train_begin(self, logs):
		self.train_start = timeit.default_timer()
		print('Training for {} epochs ...'.format(self.params['nb_epoch']))

	def on_train_end(self, logs):
		duration = timeit.default_timer() - self.train_start
		print('done, took {0:.3f} seconds'.format(duration))

	def on_epoch_begin(self, epoch, logs):
		self.epoch_start[epoch] = timeit.default_timer()
		self.inputs[epoch] = []
		self.rewards[epoch] = []
		self.actions[epoch] = []
		self.losses[epoch] = []
		self.average_qs[epoch] = []

	def on_epoch_end(self, epoch, logs):
		if len(self.average_qs[epoch]) == 0:
			self.average_qs[epoch].append(0.)
		if len(self.losses[epoch]) == 0:
			self.losses[epoch].append(0.)
		duration = timeit.default_timer() - self.epoch_start[epoch]
		steps = len(self.inputs[epoch])
		template = 'Epoch {0}: {1:.3f}s, steps: {2}, steps per second: {3:.0f}, reward: {4:.3f} [{5:.3f}, {6:.3f}], loss: {7:.3f}, avg q: {14:.3f} action: {8:.3f} [{9:.3f}, {10:.3f}], inputs: {11:.3f} [{12:.3f}, {13:.3f}]'
		variables = [
			epoch,
			duration,
			steps,
			float(steps) / duration,
			np.sum(self.rewards[epoch]),
			np.min(self.rewards[epoch]),
			np.max(self.rewards[epoch]),
			np.sum(self.losses[epoch]),
			np.mean(self.actions[epoch]),
			np.min(self.actions[epoch]),
			np.max(self.actions[epoch]),
			np.mean(self.inputs[epoch]),
			np.min(self.inputs[epoch]),
			np.max(self.inputs[epoch]),
			np.mean(self.average_qs[epoch]),
		]
		print(template.format(*variables))

		# Free up resources.
		del self.epoch_start[epoch]
		del self.inputs[epoch]
		del self.rewards[epoch]
		del self.actions[epoch]
		del self.losses[epoch]
		del self.average_qs[epoch]

	def on_batch_end(self, batch, logs):
		epoch = logs['epoch']
		self.inputs[epoch].append(np.concatenate(logs['input']))
		self.rewards[epoch].append(logs['reward'])
		self.actions[epoch].append(logs['action'])
		
		stats = logs['stats']
		if not isinstance(stats, (list, tuple)):
			stats = [stats]
		self.losses[epoch].append(stats[0])
		if len(stats) >= 2:
			self.average_qs[epoch].append(stats[1])

class TrainIntervalLogger(Callback):
	def __init__(self, interval=10000):
		self.interval = interval
		self.steps = 0
		self.reset()

	def reset(self):
		self.interval_start = timeit.default_timer()
		self.inputs = []
		self.rewards = []
		self.actions = []
		self.losses = []
		self.average_qs = []

	def on_train_begin(self, logs):
		self.train_start = timeit.default_timer()
		print('Training for {} epochs ...'.format(self.params['nb_epoch']))

	def on_train_end(self, logs):
		duration = timeit.default_timer() - self.train_start
		print('done, took {0:.3f} seconds'.format(duration))

	def print_report(self):
		if len(self.average_qs) == 0:
			self.average_qs.append(0.)
		if len(self.losses) == 0:
			self.losses.append(0.)
		duration = timeit.default_timer() - self.interval_start
		interval_steps = len(self.inputs)
		template = '{0}: {1:.3f}s, interval steps: {2}, steps per second: {3:.0f}, reward: {4:.3f} [{5:.3f}, {6:.3f}], loss: {7:.3f}, avg q: {14:.3f} action: {8:.3f} [{9:.3f}, {10:.3f}], inputs: {11:.3f} [{12:.3f}, {13:.3f}]'
		variables = [
			self.steps,
			duration,
			interval_steps,
			float(interval_steps) / duration,
			np.sum(self.rewards),
			np.min(self.rewards),
			np.max(self.rewards),
			np.sum(self.losses),
			np.mean(self.actions),
			np.min(self.actions),
			np.max(self.actions),
			np.mean(self.inputs),
			np.min(self.inputs),
			np.max(self.inputs),
			np.mean(self.average_qs),
		]
		print(template.format(*variables))

	def on_batch_end(self, batch, logs):
		if self.steps % self.interval == 0:
			if self.steps > 0:
				self.print_report()
			self.reset()

		# Book-keeping.
		self.steps += 1
		self.inputs.append(np.concatenate(logs['input']))
		self.rewards.append(logs['reward'])
		self.actions.append(logs['action'])
		stats = logs['stats']
		if not isinstance(stats, (list, tuple)):
			stats = [stats]
		self.losses.append(stats[0])
		if len(stats) >= 2:
			self.average_qs.append(stats[1])
