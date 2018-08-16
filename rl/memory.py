from __future__ import absolute_import
from collections import deque, namedtuple
import warnings
import random
import numpy as np
from rl.util import SumSegmentTree, MinSegmentTree

# This is to be understood as a transition: Given `state0`, performing `action`
# yields `reward` and results in `state1`, which might be `terminal`.
Experience = namedtuple('Experience', 'state0, action, reward, state1, terminal1')


def sample_batch_indexes(low, high, size):
    """Return a sample of (size) unique elements between low and high
        # Argument
            low (int): The minimum value for our samples
            high (int): The maximum value for our samples
            size (int): The number of samples to pick
        # Returns
            A list of samples of length size, with values between low and high
        """
    if high - low >= size:
        # We have enough data. Draw without replacement, that is each index is unique in the
        # batch. We cannot use `np.random.choice` here because it is horribly inefficient as
        # the memory grows. See https://github.com/numpy/numpy/issues/2764 for a discussion.
        # `random.sample` does the same thing (drawing without replacement) and is way faster.
        try:
            r = xrange(low, high)
        except NameError:
            r = range(low, high)
        batch_idxs = random.sample(r, size)
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
        """Return element of buffer at specific index
        # Argument
            idx (int): Index wanted
        # Returns
            The element of buffer at given index
        """
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def append(self, v):
        """Append an element to the buffer
        # Argument
            v (object): Element to append
        """
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


def zeroed_observation(observation):
    """Return an array of zeros with same shape as given observation
    # Argument
        observation (list): List of observation
    # Return
        A np.ndarray of zeros with observation.shape
    """
    if hasattr(observation, 'shape'):
        return np.zeros(observation.shape)
    elif hasattr(observation, '__iter__'):
        out = []
        for x in observation:
            out.append(zeroed_observation(x))
        return out
    else:
        return 0.


class Memory(object):
    def __init__(self, window_length, ignore_episode_boundaries=False):
        self.window_length = window_length
        self.ignore_episode_boundaries = ignore_episode_boundaries

        self.recent_observations = deque(maxlen=window_length)
        self.recent_terminals = deque(maxlen=window_length)

    def sample(self, batch_size, batch_idxs=None):
        raise NotImplementedError()

    def append(self, observation, action, reward, terminal, training=True):
        self.recent_observations.append(observation)
        self.recent_terminals.append(terminal)

    def get_recent_state(self, current_observation):
        """Return list of last observations
        # Argument
            current_observation (object): Last observation
        # Returns
            A list of the last observations
        """
        # This code is slightly complicated by the fact that subsequent observations might be
        # from different episodes. We ensure that an experience never spans multiple episodes.
        # This is probably not that important in practice but it seems cleaner.
        state = [current_observation]
        idx = len(self.recent_observations) - 1
        for offset in range(0, self.window_length - 1):
            current_idx = idx - offset
            current_terminal = self.recent_terminals[current_idx - 1] if current_idx - 1 >= 0 else False
            if current_idx < 0 or (not self.ignore_episode_boundaries and current_terminal):
                # The previously handled observation was terminal, don't add the current one.
                # Otherwise we would leak into a different episode.
                break
            state.insert(0, self.recent_observations[current_idx])
        while len(state) < self.window_length:
            state.insert(0, zeroed_observation(state[0]))
        return state

    def get_config(self):
        """Return configuration (window_length, ignore_episode_boundaries) for Memory
        # Return
            A dict with keys window_length and ignore_episode_boundaries
        """
        config = {
            'window_length': self.window_length,
            'ignore_episode_boundaries': self.ignore_episode_boundaries,
        }
        return config

class SequentialMemory(Memory):
    def __init__(self, limit, **kwargs):
        super(SequentialMemory, self).__init__(**kwargs)

        self.limit = limit

        # Do not use deque to implement the memory. This data structure may seem convenient but
        # it is way too slow on random access. Instead, we use our own ring buffer implementation.
        self.actions = RingBuffer(limit)
        self.rewards = RingBuffer(limit)
        self.terminals = RingBuffer(limit)
        self.observations = RingBuffer(limit)

    def sample(self, batch_size, batch_idxs=None):
        """Return a randomized batch of experiences
        # Argument
            batch_size (int): Size of the all batch
            batch_idxs (int): Indexes to extract
        # Returns
            A list of experiences randomly selected
        """
        # It is not possible to tell whether the first state in the memory is terminal, because it
        # would require access to the "terminal" flag associated to the previous state. As a result
        # we will never return this first state (only using `self.terminals[0]` to know whether the
        # second state is terminal).
        # In addition we need enough entries to fill the desired window length.
        assert self.nb_entries >= self.window_length + 2, 'not enough entries in the memory'

        if batch_idxs is None:
            # Draw random indexes such that we have enough entries before each index to fill the
            # desired window length.
            batch_idxs = sample_batch_indexes(
                self.window_length, self.nb_entries - 1, size=batch_size)
        batch_idxs = np.array(batch_idxs) + 1
        assert np.min(batch_idxs) >= self.window_length + 1
        assert np.max(batch_idxs) < self.nb_entries
        assert len(batch_idxs) == batch_size

        # Create experiences
        experiences = []
        for idx in batch_idxs:
            terminal0 = self.terminals[idx - 2]
            while terminal0:
                # Skip this transition because the environment was reset here. Select a new, random
                # transition and use this instead. This may cause the batch to contain the same
                # transition twice.
                idx = sample_batch_indexes(self.window_length + 1, self.nb_entries, size=1)[0]
                terminal0 = self.terminals[idx - 2]
            assert self.window_length + 1 <= idx < self.nb_entries

            # This code is slightly complicated by the fact that subsequent observations might be
            # from different episodes. We ensure that an experience never spans multiple episodes.
            # This is probably not that important in practice but it seems cleaner.
            state0 = [self.observations[idx - 1]]
            for offset in range(0, self.window_length - 1):
                current_idx = idx - 2 - offset
                assert current_idx >= 1
                current_terminal = self.terminals[current_idx - 1]
                if current_terminal and not self.ignore_episode_boundaries:
                    # The previously handled observation was terminal, don't add the current one.
                    # Otherwise we would leak into a different episode.
                    break
                state0.insert(0, self.observations[current_idx])
            while len(state0) < self.window_length:
                state0.insert(0, zeroed_observation(state0[0]))
            action = self.actions[idx - 1]
            reward = self.rewards[idx - 1]
            terminal1 = self.terminals[idx - 1]

            # Okay, now we need to create the follow-up state. This is state0 shifted on timestep
            # to the right. Again, we need to be careful to not include an observation from the next
            # episode if the last state is terminal.
            state1 = [np.copy(x) for x in state0[1:]]
            state1.append(self.observations[idx])

            assert len(state0) == self.window_length
            assert len(state1) == len(state0)
            experiences.append(Experience(state0=state0, action=action, reward=reward,
                                          state1=state1, terminal1=terminal1))
        assert len(experiences) == batch_size
        return experiences

    def append(self, observation, action, reward, terminal, training=True):
        """Append an observation to the memory
        # Argument
            observation (dict): Observation returned by environment
            action (int): Action taken to obtain this observation
            reward (float): Reward obtained by taking this action
            terminal (boolean): Is the state terminal
        """
        super(SequentialMemory, self).append(observation, action, reward, terminal, training=training)

        # This needs to be understood as follows: in `observation`, take `action`, obtain `reward`
        # and weather the next state is `terminal` or not.
        if training:
            self.observations.append(observation)
            self.actions.append(action)
            self.rewards.append(reward)
            self.terminals.append(terminal)

    @property
    def nb_entries(self):
        """Return number of observations
        # Returns
            Number of observations
        """
        return len(self.observations)

    def get_config(self):
        """Return configurations of SequentialMemory
        # Returns
            Dict of config
        """
        config = super(SequentialMemory, self).get_config()
        config['limit'] = self.limit
        return config



class EpisodeParameterMemory(Memory):
    def __init__(self, limit, **kwargs):
        super(EpisodeParameterMemory, self).__init__(**kwargs)
        self.limit = limit

        self.params = RingBuffer(limit)
        self.intermediate_rewards = []
        self.total_rewards = RingBuffer(limit)

    def sample(self, batch_size, batch_idxs=None):
        """Return a randomized batch of params and rewards
        # Argument
            batch_size (int): Size of the all batch
            batch_idxs (int): Indexes to extract
        # Returns
            A list of params randomly selected and a list of associated rewards
        """
        if batch_idxs is None:
            batch_idxs = sample_batch_indexes(0, self.nb_entries, size=batch_size)
        assert len(batch_idxs) == batch_size

        batch_params = []
        batch_total_rewards = []
        for idx in batch_idxs:
            batch_params.append(self.params[idx])
            batch_total_rewards.append(self.total_rewards[idx])
        return batch_params, batch_total_rewards

    def append(self, observation, action, reward, terminal, training=True):
        """Append a reward to the memory
        # Argument
            observation (dict): Observation returned by environment
            action (int): Action taken to obtain this observation
            reward (float): Reward obtained by taking this action
            terminal (boolean): Is the state terminal
        """
        super(EpisodeParameterMemory, self).append(observation, action, reward, terminal, training=training)
        if training:
            self.intermediate_rewards.append(reward)

    def finalize_episode(self, params):
        """Append an observation to the memory
        # Argument
            observation (dict): Observation returned by environment
            action (int): Action taken to obtain this observation
            reward (float): Reward obtained by taking this action
            terminal (boolean): Is the state terminal
        """
        total_reward = sum(self.intermediate_rewards)
        self.total_rewards.append(total_reward)
        self.params.append(params)
        self.intermediate_rewards = []

    @property
    def nb_entries(self):
        """Return number of episode rewards
        # Returns
            Number of episode rewards
        """
        return len(self.total_rewards)

    def get_config(self):
        """Return configurations of SequentialMemory
        # Returns
            Dict of config
        """
        config = super(SequentialMemory, self).get_config()
        config['limit'] = self.limit
        return config

class PartitionedRingBuffer(object):
    """
    Buffer with a section that can be sampled from but never overwritten.
    Used for demonstration data (DQfD). Can be used without a partition,
    where it would function as a fixed-idxs variant of RingBuffer.
    """
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.length = 0
        self.data = [None for _ in range(maxlen)]
        self.permanent_idx = 0
        self.next_idx = 0

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0:
            raise KeyError()
        return self.data[idx % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            self.length += 1
        self.data[(self.permanent_idx + self.next_idx)] = v
        self.next_idx = (self.next_idx + 1) % (self.maxlen - self.permanent_idx)

    def load(self, load_data):
        assert len(load_data) < self.maxlen, "Must leave space to write new data."
        for idx, data in enumerate(load_data):
            self.length += 1
            self.data[idx] = data
            self.permanent_idx += 1

class PrioritizedMemory(Memory):
    def __init__(self, limit, alpha=.4, start_beta=1., end_beta=1., steps_annealed=1, **kwargs):
        super(PrioritizedMemory, self).__init__(**kwargs)

        #The capacity of the replay buffer
        self.limit = limit

        #Transitions are stored in individual RingBuffers, similar to the SequentialMemory.
        self.actions = PartitionedRingBuffer(limit)
        self.rewards = PartitionedRingBuffer(limit)
        self.terminals = PartitionedRingBuffer(limit)
        self.observations = PartitionedRingBuffer(limit)

        assert alpha >= 0
        #how aggressively to sample based on TD error
        self.alpha = alpha
        #how aggressively to compensate for that sampling. This value is typically annealed
        #to stabilize training as the model converges (beta of 1.0 fully compensates for TD-prioritized sampling).
        self.start_beta = start_beta
        self.end_beta = end_beta
        self.steps_annealed = steps_annealed

        #SegmentTrees need a leaf count that is a power of 2
        tree_capacity = 1
        while tree_capacity < self.limit:
            tree_capacity *= 2

        #Create SegmentTrees with this capacity
        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        self.max_priority = 1.

        #wrapping index for interacting with the trees
        self.next_index = 0

    def append(self, observation, action, reward, terminal, training=True):\
        #super() call adds to the deques that hold the most recent info, which is fed to the agent
        #on agent.forward()
        super(PrioritizedMemory, self).append(observation, action, reward, terminal, training=training)
        if training:
            self.observations.append(observation)
            self.actions.append(action)
            self.rewards.append(reward)
            self.terminals.append(terminal)
            #The priority of each new transition is set to the maximum
            self.sum_tree[self.next_index] = self.max_priority ** self.alpha
            self.min_tree[self.next_index] = self.max_priority ** self.alpha

            #shift tree pointer index to keep it in sync with RingBuffers
            self.next_index = (self.next_index + 1) % self.limit

    def _sample_proportional(self, batch_size):
        #outputs a list of idxs to sample, based on their priorities.
        idxs = list()

        for _ in range(batch_size):
            mass = random.random() * self.sum_tree.sum(0, self.limit - 1)
            idx = self.sum_tree.find_prefixsum_idx(mass)
            idxs.append(idx)

        return idxs

    def sample(self, batch_size, beta=1.):
        idxs = self._sample_proportional(batch_size)

        #importance sampling weights are a stability measure
        importance_weights = list()

        #The lowest-priority experience defines the maximum importance sampling weight
        prob_min = self.min_tree.min() / self.sum_tree.sum()
        max_importance_weight = (prob_min * self.nb_entries)  ** (-beta)
        obs_t, act_t, rews, obs_t1, dones = [], [], [], [], []

        experiences = list()
        for idx in idxs:
            while idx < self.window_length + 1:
                idx += 1

            terminal0 = self.terminals[idx - 2]
            while terminal0:
                # Skip this transition because the environment was reset here. Select a new, random
                # transition and use this instead. This may cause the batch to contain the same
                # transition twice.
                idx = sample_batch_indexes(self.window_length + 1, self.nb_entries, size=1)[0]
                terminal0 = self.terminals[idx - 2]

            assert self.window_length + 1 <= idx < self.nb_entries

            #probability of sampling transition is the priority of the transition over the sum of all priorities
            prob_sample = self.sum_tree[idx] / self.sum_tree.sum()
            importance_weight = (prob_sample * self.nb_entries) ** (-beta)
            #normalize weights according to the maximum value
            importance_weights.append(importance_weight/max_importance_weight)

            # Code for assembling stacks of observations and dealing with episode boundaries is borrowed from
            # SequentialMemory
            state0 = [self.observations[idx - 1]]
            for offset in range(0, self.window_length - 1):
                current_idx = idx - 2 - offset
                assert current_idx >= 1
                current_terminal = self.terminals[current_idx - 1]
                if current_terminal and not self.ignore_episode_boundaries:
                    # The previously handled observation was terminal, don't add the current one.
                    # Otherwise we would leak into a different episode.
                    break
                state0.insert(0, self.observations[current_idx])
            while len(state0) < self.window_length:
                state0.insert(0, zeroed_observation(state0[0]))
            action = self.actions[idx - 1]
            reward = self.rewards[idx - 1]
            terminal1 = self.terminals[idx - 1]
            state1 = [np.copy(x) for x in state0[1:]]
            state1.append(self.observations[idx])

            assert len(state0) == self.window_length
            assert len(state1) == len(state0)
            experiences.append(Experience(state0=state0, action=action, reward=reward,
                                          state1=state1, terminal1=terminal1))
        assert len(experiences) == batch_size

        # Return a tuple whre the first batch_size items are the transititions
        # while -2 is the importance weights of those transitions and -1 is
        # the idxs of the buffer (so that we can update priorities later)
        return tuple(list(experiences)+ [importance_weights, idxs])

    def update_priorities(self, idxs, priorities):
        #adjust priorities based on new TD error
        for i, idx in enumerate(idxs):
            assert 0 <= idx < self.limit
            priority = priorities[i] ** self.alpha
            self.sum_tree[idx] = priority
            self.min_tree[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def calculate_beta(self, current_step):
        a = float(self.end_beta - self.start_beta) / float(self.steps_annealed)
        b = float(self.start_beta)
        current_beta = min(self.end_beta, a * float(current_step) + b)
        return current_beta

    def get_config(self):
        config = super(PrioritizedMemory, self).get_config()
        config['alpha'] = self.alpha
        config['start_beta'] = self.start_beta
        config['end_beta'] = self.end_beta
        config['beta_steps_annealed'] = self.steps_annealed

    @property
    def nb_entries(self):
        """Return number of observations
        # Returns
            Number of observations
        """
        return len(self.observations)


class PartitionedMemory(Memory):
    def __init__(self, limit, pre_load_data, alpha=.4, start_beta=1., end_beta=1., steps_annealed=1, **kwargs):
        super(PartitionedMemory, self).__init__(**kwargs)

        #The capacity of the replay buffer
        self.limit = limit

        #Transitions are stored in individual PartitionedRingBuffers.
        self.actions = PartitionedRingBuffer(limit)
        self.rewards = PartitionedRingBuffer(limit)
        self.terminals = PartitionedRingBuffer(limit)
        self.observations = PartitionedRingBuffer(limit)

        assert alpha >= 0
        #how aggressively to sample based on TD error
        self.alpha = alpha
        #how aggressively to compensate for that sampling.
        self.start_beta = start_beta
        self.end_beta = end_beta
        self.steps_annealed = steps_annealed

        #SegmentTrees need a leaf count that is a power of 2
        tree_capacity = 1
        while tree_capacity < self.limit:
            tree_capacity *= 2

        #Create SegmentTrees with this capacity
        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        self.max_priority = 1.

        #unpack the expert transitions (assumes order recorded by the rl.utils.record_demo_data() method)
        demo_obs, demo_acts, demo_rews, demo_ts = [], [], [], []
        self.pre_load_data = pre_load_data
        for demo in self.pre_load_data:
            demo_obs.append(demo[0])
            demo_acts.append(demo[1])
            demo_rews.append(demo[2])
            demo_ts.append(demo[3])

        #pre-load the demonstration data
        self.observations.load(demo_obs)
        self.actions.load(demo_acts)
        self.rewards.load(demo_rews)
        self.terminals.load(demo_ts)

        self.permanent_idx = self.observations.permanent_idx
        assert self.permanent_idx == self.rewards.permanent_idx

        self.next_index = 0

        for idx in range(self.permanent_idx):
            self.sum_tree[idx] = (self.max_priority ** self.alpha)
            self.min_tree[idx] = (self.max_priority ** self.alpha)

    def append(self, observation, action, reward, terminal, training=True):
        #super() call adds to the deques that hold the most recent info, which is fed to the agent
        #on agent.forward()
        super(PartitionedMemory, self).append(observation, action, reward, terminal, training=training)
        if training:
            self.observations.append(observation)
            self.actions.append(action)
            self.rewards.append(reward)
            self.terminals.append(terminal)
            #The priority of each new transition is set to the maximum
            self.sum_tree[self.next_index + self.permanent_idx] = self.max_priority ** self.alpha
            self.min_tree[self.next_index + self.permanent_idx] = self.max_priority ** self.alpha
            #shift tree pointer index to keep it in sync with RingBuffers
            self.next_index = ((self.next_index + 1) % (self.limit - self.permanent_idx))

    def sample_proportional(self, batch_size):
        """
        Outputs a list of idxs to sample, based on their priorities.

        This function is public in this memory (vs. private in Sequential and
        Prioritized), because DQfD needs to be able to sample the same idxs
        twice (single step and n-step).
        """
        idxs = list()

        for _ in range(batch_size):
            mass = random.random() * self.sum_tree.sum(0, self.limit - 1)
            idx = self.sum_tree.find_prefixsum_idx(mass)
            idxs.append(idx)

        return idxs

    def sample_by_idxs(self, idxs, batch_size, beta=1., nstep=1, gamma=1):
        """
        Gathers transition data from the ring buffers. The PartitionedMemory
        separates generating the idxs and returning their transitions, allowing
        this method to be called multiple times with the same idxs.
        """
        #importance sampling weights are a stability measure
        importance_weights = list()

        #The lowest-priority experience defines the maximum importance sampling weight
        prob_min = self.min_tree.min() / self.sum_tree.sum()
        max_importance_weight = (prob_min * self.nb_entries)  ** (-beta)
        obs_t, act_t, rews, obs_t1, dones = [], [], [], [], []

        experiences = list()
        for idx in idxs:
            while idx < self.window_length + 1:
                idx += 1
            while idx + nstep > self.nb_entries and self.nb_entries < self.limit:
                # We are fine with nstep spilling back to the beginning of the buffer
                # once it has been filled.
                idx -= 1
            terminal0 = self.terminals[idx - 2]
            while terminal0:
                # Skip this transition because the environment was reset here. Select a new, random
                # transition and use this instead. This may cause the batch to contain the same
                # transition twice.
                idx = sample_batch_indexes(self.window_length + 1, self.nb_entries - nstep, size=1)[0]
                terminal0 = self.terminals[idx - 2]

            assert self.window_length + 1 <= idx < self.nb_entries

            #probability of sampling transition is the priority of the transition over the sum of all priorities
            prob_sample = self.sum_tree[idx] / self.sum_tree.sum()
            importance_weight = (prob_sample * self.nb_entries) ** (-beta)
            #normalize weights according to the maximum value
            importance_weights.append(importance_weight/max_importance_weight)

            #assemble the initial state from the ringbuffer.
            state0 = [self.observations[idx - 1]]
            for offset in range(0, self.window_length - 1):
                current_idx = idx - 2 - offset
                assert current_idx >= 1
                current_terminal = self.terminals[current_idx - 1]
                if current_terminal and not self.ignore_episode_boundaries:
                    # The previously handled observation was terminal, don't add the current one.
                    # Otherwise we would leak into a different episode.
                    break
                state0.insert(0, self.observations[current_idx])
            while len(state0) < self.window_length:
                state0.insert(0, zeroed_observation(state0[0]))

            action = self.actions[idx - 1]
            # N-step TD
            reward = 0
            nstep = nstep
            for i in range(nstep):
                reward += (gamma**i) * self.rewards[idx + i - 1]
                if self.terminals[idx + i - 1]:
                    #episode terminated before length of n-step rollout.
                    nstep = i
                    break

            terminal1 = self.terminals[idx + nstep - 1]

            # We assemble the second state in a similar way.
            state1 = [self.observations[idx + nstep - 1]]
            for offset in range(0, self.window_length - 1):
                current_idx = idx + nstep - 1 - offset
                assert current_idx >= 1
                current_terminal = self.terminals[current_idx - 1]
                if current_terminal and not self.ignore_episode_boundaries:
                    # The previously handled observation was terminal, don't add the current one.
                    # Otherwise we would leak into a different episode.
                    break
                state1.insert(0, self.observations[current_idx])
            while len(state1) < self.window_length:
                state1.insert(0, zeroed_observation(state0[0]))

            assert len(state0) == self.window_length
            assert len(state1) == len(state0)
            experiences.append(Experience(state0=state0, action=action, reward=reward,
                                          state1=state1, terminal1=terminal1))
        assert len(experiences) == batch_size

        return tuple(list(experiences)+ [importance_weights, idxs])

    def update_priorities(self, idxs, priorities):
        #adjust priorities based on new TD error
        for i, idx in enumerate(idxs):
            assert 0 <= idx < self.limit
            #expert transition priorities receive an extra boost
            if idx < self.permanent_idx:
                priority = (priorities[i] ** self.alpha) + .999
            else:
                priority = (priorities[i] ** self.alpha)
            self.sum_tree[idx] = priority
            self.min_tree[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def calculate_beta(self, current_step):
        a = float(self.end_beta - self.start_beta) / float(self.steps_annealed)
        b = float(self.start_beta)
        current_beta = min(self.end_beta, a * float(current_step) + b)
        return current_beta

    def get_config(self):
        config = super(PartitionedMemory, self).get_config()
        config['alpha'] = self.alpha
        config['start_beta'] = self.start_beta
        config['end_beta'] = self.end_beta
        config['beta_steps_annealed'] = self.steps_annealed
        config['pre_load_data'] = self.pre_load_data

    @property
    def nb_entries(self):
        """Return number of observations
        # Returns
            Number of observations
        """
        return len(self.observations)
