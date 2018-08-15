# This memory specifically written for ACER agent.
# For Sequential memory please visit memory.py in the rl folder.

from collections import deque, namedtuple
import numpy as np


class EpisodeMemory(object):
    def __init__(self, nsteps, size=50000, nenv=1):
        self.nenv = nenv
        self.nsteps = nsteps
        self.maxlen = size//nsteps
        self.memory = deque(maxlen=self.maxlen)

    def put(self, trajectory):
        self.memory.append(trajectory)

    def get(self, batch_size=4):
        l = self.length()
        assert l>batch_size, "Not enough experience stored. Store more samples"
        idx_batch = np.random.choice(len(self.memory), batch_size)
        batch = [val for i, val in enumerate(self.memory) if i in idx_batch] 
        return batch
        
    def length(self):
        return len(self.memory)