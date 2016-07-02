from keras.layers.normalization import BatchNormalization
from keras import initializations
from keras import backend as K


class StatefulBatchNormalization(BatchNormalization):
	def __init__(self, epsilon=1e-6, mode=0, axis=-1, momentum=0.9,
				 weights=None, beta_init='zero', gamma_init='one', stateful=False, **kwargs):
		super(StatefulBatchNormalization, self).__init__(**kwargs)
		self.stateful = stateful

	def get_output(self, train):
		X = self.get_input(train)
		if self.mode == 0:
			input_shape = self.input_shape
			reduction_axes = list(range(len(input_shape)))
			del reduction_axes[self.axis]
			broadcast_shape = [1] * len(input_shape)
			broadcast_shape[self.axis] = input_shape[self.axis]

			if train or self.stateful:
				m = K.mean(X, axis=reduction_axes)
				brodcast_m = K.reshape(m, broadcast_shape)
				std = K.mean(K.square(X - brodcast_m) + self.epsilon, axis=reduction_axes)
				std = K.sqrt(std)
				brodcast_std = K.reshape(std, broadcast_shape)
				mean_update = self.momentum * self.running_mean + (1-self.momentum) * m
				std_update = self.momentum * self.running_std + (1-self.momentum) * std
				self.updates = [(self.running_mean, mean_update),
								(self.running_std, std_update)]

			if train:
				X_normed = (X - brodcast_m) / (brodcast_std + self.epsilon)
			else:
				brodcast_m = K.reshape(self.running_mean, broadcast_shape)
				brodcast_std = K.reshape(self.running_std, broadcast_shape)
				X_normed = ((X - brodcast_m) /
							(brodcast_std + self.epsilon))
			out = K.reshape(self.gamma, broadcast_shape) * X_normed + K.reshape(self.beta, broadcast_shape)
		elif self.mode == 1:
			m = K.mean(X, axis=-1, keepdims=True)
			std = K.std(X, axis=-1, keepdims=True)
			X_normed = (X - m) / (std + self.epsilon)
			out = self.gamma * X_normed + self.beta
		return out

	def get_config(self):
		config = {'stateful': self.stateful}
		base_config = super(StatefulBatchNormalization, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))
