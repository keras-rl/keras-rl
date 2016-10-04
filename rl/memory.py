from __future__ import absolute_import
from collections import deque, namedtuple
import warnings
import random

import numpy as np


# This is to be understood as a transition: Given `state0`, performing `action`
# yields `reward` and results in `state1`, which might be `terminal`.
Experience = namedtuple('Experience', 'state0, action, reward, terminal, state1')


def sample_batch_indexes(low, high, size):
    if high - low >= size:
        # We have enough data. Draw without replacement, that is each index is unique in the
        # batch. We cannot use `np.random.choice` here because it is horribly inefficient as
        # the memory grows. See https://github.com/numpy/numpy/issues/2764 for a discussion.
        # `random.sample` does the same thing (drawing without replacement) and is way faster.
        batch_idxs = random.sample(xrange(low, high), size)
    else:
        # Not enough data. Help ourselves with sampling from the range, but the same index
        # can occur multiple times. This is not good and should be avoided by picking a
        # large enough warm-up phase.
        warnings.warn('Not enough entries to sample without replacement. Consider increasing your warm-up phase to avoid oversampling!')
        batch_idxs = np.random.random_integers(low, high - 1, size=size)
    assert len(batch_idxs) == size
    return batch_idxs


class RingBuffer(object):
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = [None for _ in range(maxlen)]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v

class SequentialMemory(object):
    def __init__(self, limit, ignore_episode_boundaries=False):
        self.limit = limit
        self.ignore_episode_boundaries = ignore_episode_boundaries

        # Do not use deque to implement the memory. This data structure may seem convenient but
        # it is way too slow on random access. Instead, we use our own ring buffer implementation.
        self.actions = RingBuffer(limit)
        self.rewards = RingBuffer(limit)
        self.terminals = RingBuffer(limit)
        self.observations = RingBuffer(limit)

    def sample(self, batch_size, window_length, batch_idxs=None):
        if batch_idxs is None:
            # Draw random indexes such that we have at least `window_length` entries before each index.
            batch_idxs = sample_batch_indexes(window_length, self.nb_entries, size=batch_size)
        assert len(batch_idxs) == batch_size

        # Create experiences
        experiences = []
        for idx in batch_idxs:
            # This code is slightly complicated by the fact that subsequent observations might be
            # from different episodes. We ensure that an experience never spans multiple episodes.
            # This is probably not that important in practice but it seems cleaner.
            state0 = [self.observations[idx - 1]]
            for current_idx in range(idx - 2, idx - window_length - 1, -1):
                if not self.ignore_episode_boundaries and self.terminals[current_idx]:
                    break
                state0.insert(0, self.observations[current_idx])
            while len(state0) < window_length:
                state0.insert(0, np.zeros(state0[0].shape))

            action = self.actions[idx - 1]
            reward = self.rewards[idx - 1]
            terminal = self.terminals[idx - 1]

            # Okay, now we need to create the follow-up state. This is state0 shifted on timestep
            # to the right. Again, we need to be careful to not include an observation from the next
            # episode if the last state is terminal.
            state1 = [np.copy(x) for x in state0[1:]]
            if not self.ignore_episode_boundaries and terminal:
                state1.append(np.zeros(state0[-1].shape))
            else:
                state1.append(self.observations[idx])

            assert len(state0) == window_length
            assert len(state1) == len(state0)
            experiences.append(Experience(state0, action, reward, terminal, state1))
        assert len(experiences) == batch_size
        return experiences

    def get_recent_state(self, current_observation, window_length):
        # This code is slightly complicated by the fact that subsequent observations might be
        # from different episodes. We ensure that an experience never spans multiple episodes.
        # This is probably not that important in practice but it seems cleaner.
        state = [current_observation]
        idx = self.nb_entries - 1
        for offset in range(0, window_length - 1):
            current_idx = idx - offset
            if current_idx < 0 or (not self.ignore_episode_boundaries and self.terminals[current_idx]):
                break
            state.insert(0, self.observations[current_idx])
        while len(state) < window_length:
            state.insert(0, np.zeros(state[0].shape))
        state = np.array(state)
        assert state.shape[0] == window_length
        return state

    def append(self, observation, action, reward, terminal):
        # This needs to be understood as follows: in `observation`, take `action`, obtain `reward`
        # and weather the next state is `terminal` or not.
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.terminals.append(terminal)

    @property
    def nb_entries(self):
        return len(self.observations)


class EpisodeParameterMemory(object):
    def __init__(self,limit,max_episode_steps):
        self.limit = limit
        self.max_episode_steps = max_episode_steps

        self.params = RingBuffer(limit)
        self.intermediate_rewards = RingBuffer(self.max_episode_steps)
        self.reward_totals = RingBuffer(limit)

    def sample(self, batch_size, batch_idxs=None):
        if batch_idxs is None:
            batch_idxs = sample_batch_indexes(0, self.nb_entries, size=batch_size)
        assert len(batch_idxs) == batch_size

        params = []
        reward_totals = []
        for idx in batch_idxs:
            params.append(self.params[idx])
            reward_totals.append(self.reward_totals[idx])
        return params, reward_totals

    def append(self,reward):
        self.intermediate_rewards.append(reward)

    def finalise_episode(self,params):
        total_reward = sum([d for d in self.intermediate_rewards.data if d])
        self.reward_totals.append(total_reward)
        self.params.append(params)
        self.intermediate_rewards = RingBuffer(self.max_episode_steps)

    @property
    def nb_entries(self):
        return len(self.reward_totals)
