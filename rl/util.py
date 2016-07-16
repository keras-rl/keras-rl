from keras.models import model_from_config, Sequential, Model
import keras.optimizers as optimizers


def clone_model(model, custom_objects={}):
	config = model.get_config()
	try:
		clone = Sequential.from_config(config, custom_objects)
	except:
		clone = Model.from_config(config, custom_objects)
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
