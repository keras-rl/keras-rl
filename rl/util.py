import numpy as np

from keras.models import model_from_config, Sequential, Model, model_from_config
import keras.optimizers as optimizers
import keras.backend as K
from keras import __version__ as keras_version

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


class AbstractAdditionalUpdatesOptimizer(optimizers.Optimizer):

    def __init__(self, optimizer, additional_updates):
        super(AbstractAdditionalUpdatesOptimizer, self).__init__()
        self.optimizer = optimizer
        self.additional_updates = additional_updates

    def get_config(self):
        return self.optimizer.get_config()

    def _update_updates(updates):
        updates += self.additional_updates
        self.updates = updates
        return self.updates

class AdditionalUpdatesOptimizerK1(AbstractAdditionalUpdatesOptimizer):
    def get_updates(self, params, constraints, loss):
        return self._update_updates(self.optimizer.get_updates(params, constraints, loss))

class AdditionalUpdatesOptimizerK2(AbstractAdditionalUpdatesOptimizer):
    def get_updates(self, params, constraints, loss):
        return self._update_updates(self.optimizer.get_updates(params, loss))

if int(keras_version.split('.')[0]) == 1:
    AdditionalUpdatesOptimizer = AdditionalUpdatesOptimizerK1
else:
    AdditionalUpdatesOptimizer = AdditionalUpdatesOptimizerK2


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
