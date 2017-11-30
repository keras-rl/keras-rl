import logging
import numpy as np

from keras.models import model_from_config, Sequential, Model, model_from_config
import keras.optimizers as optimizers
import keras.backend as K


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

    # Keras sometimes passes params and loss as keyword arguments,
    # expecting constraints to be optional, so there must be a default
    # value for constraints here; see for example:
    # https://github.com/fchollet/keras/blob/master/keras/engine/training.py#L988-L990
    def get_updates(self, params, constraints=None, loss=None):
        assert loss is not None, (params, constraints, loss)
        updates = self.optimizer.get_updates(params, constraints, loss)
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


def _freeze_unfreeze_layers_of_model(model, freeze=True):
    """Freezes layes so they are not changes during training. essential for transfer learning."""
    for layer_idx in range(len(model.layers)):
        if hasattr(model.layers[layer_idx], 'layers'):
            NnBase._freeze_unfreeze_layers_of_model(model.layers[layer_idx])
        else:
            model.layers[layer_idx].trainable = not freeze


def freeze_unfreeze_n_layers(model, n, freeze=True):
    """Freezes either first n layers or last n layers of a network depending upon if
    n is +ve or -ve"""
    if not model.optimizer:
        raise RuntimeError(
            'Your tried to fit your model but it hasn\'t been compiled yet. '
            'Please call `compile()`. Otherwise this is error prone.')
    if n < 0:
        n += len(model.layers)
        idx_range = range(n, len(model.layers), 1)
    else:
        idx_range = range(n)
    for i in idx_range:
        if len(model.layers[i].get_weights()) > 0:
            if freeze:
                logging.info("Freezing " + model.layers[i].name)
            else:
                logging.info("Unfreezing " + model.layers[i].name)
            if hasattr(model.layers[i], 'layers'):
                # If a model layer is itself a model then it freezes recursively
                NnBase._freeze_unfreeze_layers_of_model(model.layers[i], freeze)
            else:
                model.layers[i].trainable = not freeze

    model.compile(loss=model.loss, optimizer=model.optimizer, metrics=model.metrics)


def freeze_by_binary_flag(model, flag_list):
    """Takes a flag_list of binary values which is as long as number of layers in the model.
    Any layer with a True flag gets frozen"""
    if not model.optimizer:
        raise RuntimeError(
            'Your tried to fit your model but it hasn\'t been compiled yet. '
            'Please call `compile()`. Otherwise this is error prone.')

    if len(flag_list) != len(model.layers):
        raise RuntimeError('The number of flags provided doesn\'t match number of layers in the model  = '
                           ''+ str(len(flag_list)) + 'number of ids = ' + str(len(model.layers)))
    for i in range(len(flag_list)):
        if flag_list[i] == 1:
            if len(model.layers[i].get_weights()) <= 0:
                raise RuntimeError("Trying to freeze layer with no weights")
            else:
                logging.info("Freezing " + model.layers[i].name)

                if hasattr(model.layers[i], 'layers'):
                    # If a model layer is itself a model then it freezes recursively
                    NnBase._freeze_unfreeze_layers_of_model(model.layers[i])
                else:
                    model.layers[i].trainable = False

    model.compile(loss=model.loss, optimizer=model.optimizer, metrics=model.metrics)
