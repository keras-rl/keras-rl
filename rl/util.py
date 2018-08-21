import numpy as np
import operator
from keras.models import model_from_config, Sequential, Model, model_from_config
import keras.optimizers as optimizers
import keras.backend as K
import time

def clone_model(model, custom_objects={}):
    # Requires Keras 1.0.7 since get_config has breaking changes.
    config = {
        'class_name': model.__class__.__name__,
        'config': model.get_config(),
    }
    clone = model_from_config(config, custom_objects=custom_objects)
    clone.set_weights(model.get_weights())
    return clone


def clone_optimizer(optimizer):
    if type(optimizer) is str:
        return optimizers.get(optimizer)
    # Requires Keras 1.0.7 since get_config has breaking changes.
    params = dict([(k, v) for k, v in optimizer.get_config().items()])
    config = {
        'class_name': optimizer.__class__.__name__,
        'config': params,
    }
    if hasattr(optimizers, 'optimizer_from_config'):
        # COMPATIBILITY: Keras < 2.0
        clone = optimizers.optimizer_from_config(config)
    else:
        clone = optimizers.deserialize(config)
    return clone


def get_soft_target_model_updates(target, source, tau):
    target_weights = target.trainable_weights + sum([l.non_trainable_weights for l in target.layers], [])
    source_weights = source.trainable_weights + sum([l.non_trainable_weights for l in source.layers], [])
    assert len(target_weights) == len(source_weights)

    # Create updates.
    updates = []
    for tw, sw in zip(target_weights, source_weights):
        updates.append((tw, tau * sw + (1. - tau) * tw))
    return updates


def get_object_config(o):
    if o is None:
        return None

    config = {
        'class_name': o.__class__.__name__,
        'config': o.get_config()
    }
    return config


def huber_loss(y_true, y_pred, clip_value):
    # Huber loss, see https://en.wikipedia.org/wiki/Huber_loss and
    # https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b
    # for details.
    assert clip_value > 0.

    x = y_true - y_pred
    if np.isinf(clip_value):
        # Spacial case for infinity since Tensorflow does have problems
        # if we compare `K.abs(x) < np.inf`.
        return .5 * K.square(x)

    condition = K.abs(x) < clip_value
    squared_loss = .5 * K.square(x)
    linear_loss = clip_value * (K.abs(x) - .5 * clip_value)
    if K.backend() == 'tensorflow':
        import tensorflow as tf
        if hasattr(tf, 'select'):
            return tf.select(condition, squared_loss, linear_loss)  # condition, true, false
        else:
            return tf.where(condition, squared_loss, linear_loss)  # condition, true, false
    elif K.backend() == 'theano':
        from theano import tensor as T
        return T.switch(condition, squared_loss, linear_loss)
    else:
        raise RuntimeError('Unknown backend "{}".'.format(K.backend()))


class AdditionalUpdatesOptimizer(optimizers.Optimizer):
    def __init__(self, optimizer, additional_updates):
        super(AdditionalUpdatesOptimizer, self).__init__()
        self.optimizer = optimizer
        self.additional_updates = additional_updates

    def get_updates(self, params, loss):
        updates = self.optimizer.get_updates(params=params, loss=loss)
        updates += self.additional_updates
        self.updates = updates
        return self.updates

    def get_config(self):
        return self.optimizer.get_config()


# Based on https://github.com/openai/baselines/blob/master/baselines/common/mpi_running_mean_std.py
class WhiteningNormalizer(object):
    def __init__(self, shape, eps=1e-2, dtype=np.float64):
        self.eps = eps
        self.shape = shape
        self.dtype = dtype

        self._sum = np.zeros(shape, dtype=dtype)
        self._sumsq = np.zeros(shape, dtype=dtype)
        self._count = 0

        self.mean = np.zeros(shape, dtype=dtype)
        self.std = np.ones(shape, dtype=dtype)

    def normalize(self, x):
        return (x - self.mean) / self.std

    def denormalize(self, x):
        return self.std * x + self.mean

    def update(self, x):
        if x.ndim == len(self.shape):
            x = x.reshape(-1, *self.shape)
        assert x.shape[1:] == self.shape

        self._count += x.shape[0]
        self._sum += np.sum(x, axis=0)
        self._sumsq += np.sum(np.square(x), axis=0)

        self.mean = self._sum / float(self._count)
        self.std = np.sqrt(np.maximum(np.square(self.eps), self._sumsq / float(self._count) - np.square(self.mean)))


class SegmentTree(object):
    """
    Abstract SegmentTree data structure used to create PrioritizedMemory.
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py
    """
    def __init__(self, capacity, operation, neutral_element):

        #powers of two have no bits in common with the previous integer
        assert capacity > 0 and capacity & (capacity - 1) == 0, "Capacity must be positive and a power of 2"
        self._capacity = capacity

        #a segment tree has (2*n)-1 total nodes
        self._value = [neutral_element for _ in range(2 * capacity)]

        self._operation = operation

        self.next_index = 0

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]

class SumSegmentTree(SegmentTree):
    """
    SumTree allows us to sum priorities of transitions in order to assign each a probability of being sampled.
    """
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=operator.add,
            neutral_element=0.0
        )

    def sum(self, start=0, end=None):
        """Returns arr[start] + ... + arr[end]"""
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum
        if array values are probabilities, this function
        allows to sample indexes according to the discrete
        probability efficiently.
        Parameters
        ----------
        perfixsum: float
            upperbound on the sum of array prefix
        Returns
        -------
        idx: int
            highest index satisfying the prefixsum constraint
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity

class MinSegmentTree(SegmentTree):
    """
    In PrioritizedMemory, we normalize importance weights according to the maximum weight in the buffer.
    This is determined by the minimum transition priority. This MinTree provides an efficient way to
    calculate that.
    """
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=min,
            neutral_element=float('inf')
        )

    def min(self, start=0, end=None):
        """Returns min(arr[start], ...,  arr[end])"""

        return super(MinSegmentTree, self).reduce(start, end)

def record_demo_data(env_name, steps, frame_delay=0.03, env_seed=123, data_filepath='expert_demo_data.npy'):
    """
    Basic script for recording your own demonstration gameplay in a gym environment. Modified
    from gym keyboard agent.
    """
    import gym
    env = gym.make(env_name)
    np.random.seed(env_seed)
    env.seed(env_seed)
    nb_actions = env.action_space.n

    action = 0
    human_wants_restart = False
    human_sets_pause = False

    def key_press(key, mod):
        nonlocal action, human_sets_pause, human_wants_restart
        if key==0xff0d: human_wants_restart = True
        if key==32: human_sets_pause = not human_sets_pause
        a = int( key - ord('0') )
        if a <= 0 or a >= nb_actions: return
        action = a

    def key_release(key, mod):
        nonlocal action
        a = int( key - ord('0') )
        if a <= 0 or a >= nb_actions: return
        if action == a:
            action = 0

    env.render(mode='human')
    env.unwrapped.viewer.window.on_key_press = key_press
    env.unwrapped.viewer.window.on_key_release = key_release

    print("ACTIONS={}".format(nb_actions))
    print("Press keys 1 2 3 ... to take actions 1 2 3 ...")
    print("No keys pressed is taking action 0")

    experiences = []
    obs = env.reset()
    total_timesteps = 0

    while total_timesteps < steps:
        if total_timesteps % 1000 == 0:
            print("Steps Elapsed: " + str(total_timesteps))
        transition = [obs]
        act = action
        transition.append(act)
        obs, r, done, _ = env.step(act)
        transition.append(r)
        transition.append(done)
        experiences.append(transition)
        total_timesteps += 1
        env.render(mode='human')
        if done:
            env.reset()
        if human_wants_restart:
                transitions = []
                total_timesteps = 0
        while human_sets_pause:
            env.render(mode='human')
            time.sleep(0.1)
        #Gym runs the environments fast by default. Tweak the frame_delay parameter to adjust play speed.
        time.sleep(frame_delay)

    data_matrix = np.array(experiences)
    np.save(data_filepath, data_matrix)

def load_demo_data_from_file(data_filepath='expert_demo_data.npy'):
    return np.load(data_filepath)
