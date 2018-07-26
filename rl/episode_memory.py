import random
from collections import deque, namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'done', 'policy'))

class EpisodeMemory(object):
	def __init__(self, env, nstep, size=50000):
		self.env = env
		self.nenv = len(self.env)
		self.nsteps = nsteps
		self.maxlen = size//nsteps
		self.memory = deque(maxlen=self.maxlen)
		# self.idx = [i for i in range()]
		self.trajectory = []

	def put(self, state, action, reward, done, policy, rank):
		self.trajectory.append(Transition(state, action, reward, done, policy))
		if done:
			self.memory.append(self.trajectory)
			self.trajectory = []

	def get(self, batch_size=4, trajectory_length=20):
		l = self.length()
		assert l>batch_size, "Not enough experience stored. Store more samples"
		batch = random.sample(self.memory, batch_size)
		min_length = min(min(len(trajectory) for trajectory in batch), trajectory_length)
		for i in range(len(batch)):
			index = random.randrange(len(batch[i]) - min_length - 1)
			batch[i] = batch[i][index : index + min_length]
		return batch
		
	def length(self):
		return len(self.memory)