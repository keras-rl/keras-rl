import warnings

from keras.models import model_from_config, Sequential, Model
import keras.optimizers as optimizers


def clone_model(model, custom_objects={}):
    if len(custom_objects) > 0:
        warnings.warn('Due to an API problem with Keras, custom_objects is currently ignored. Sorry about that.')
    config = model.get_config()
    try:
        # TODO: re-enable custom_objects
        clone = Sequential.from_config(config)
    except:
        # TODO: re-enable custom_objects
        clone = Model.from_config(config)
    clone.set_weights(model.get_weights())
    return clone


def clone_optimizer(optimizer):
    params = dict([(k, v) for k, v in optimizer.get_config().items()])
    name = params.pop('name')
    clone = optimizers.get(name, params)
    if hasattr(optimizer, 'clipnorm'):
        clone.clipnorm = optimizer.clipnorm
    if hasattr(optimizer, 'clipvalue'):
        clone.clipvalue = optimizer.clipvalue
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


class AdditionalUpdatesOptimizer(optimizers.Optimizer):
    def __init__(self, optimizer, additional_updates):
        super(AdditionalUpdatesOptimizer, self).__init__()
        self.optimizer = optimizer
        self.additional_updates = additional_updates

    def get_updates(self, params, constraints, loss):
        updates = self.optimizer.get_updates(params, constraints, loss)
        updates += self.additional_updates
        self.updates = updates
        return self.updates

    def get_config(self):
        return self.optimizer.get_config()
