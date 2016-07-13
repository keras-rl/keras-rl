from keras.models import model_from_config, Sequential


def clone_model(model, custom_objects={}):
	config = model.get_config()
	# TODO: add support for Graph model
	clone = Sequential.from_config(config, custom_objects)
	clone.set_weights(model.get_weights())
	return clone
