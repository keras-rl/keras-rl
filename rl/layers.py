import numpy as np
from keras import initializers, regularizers, activations, constraints
from keras.engine.topology import Layer
import keras.backend as K

class NoisyNetDense(Layer):
    def __init__(self,
                units,
                activation=None,
                kernel_constraint=None,
                bias_constraint=None,
                kernel_regularizer=None,
                bias_regularizer=None,
                mu_initializer=None,
                sigma_initializer=None,
                **kwargs):

        super(NoisyNetDense, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.kernel_constraint = constraints.get(kernel_constraint) if kernel_constraint is not None else None
        self.bias_constraint = constriants.get(bias_constraint)if kernel_constraint is not None else None
        self.kernel_regularizer = regularizers.get(kernel_regularizer)if kernel_constraint is not None else None
        self.bias_regularizer = regularizers.get(bias_regularizer) if kernel_constraint is not None else None

    def build(self, input_shape):
        self.input_dim = input_shape[-1]

        #See section 3.2 of Fortunato et al.
        sqr_inputs = self.input_dim**(1/2)
        self.sigma_initializer = initializers.Constant(value=.5/sqr_inputs)
        self.mu_initializer = initializers.RandomUniform(minval=(-1/sqr_inputs), maxval=(1/sqr_inputs))


        self.mu_weight = self.add_weight(shape=(self.input_dim, self.units),
                                        initializer=self.mu_initializer,
                                        name='mu_weights',
                                        constraint=self.kernel_constraint,
                                        regularizer=self.kernel_regularizer)

        self.sigma_weight = self.add_weight(shape=(self.input_dim, self.units),
                                        initializer=self.sigma_initializer,
                                        name='sigma_weights',
                                        constraint=self.kernel_constraint,
                                        regularizer=self.kernel_regularizer)

        self.mu_bias = self.add_weight(shape=(self.units,),
                                        initializer=self.mu_initializer,
                                        name='mu_bias',
                                        constraint=self.bias_constraint,
                                        regularizer=self.bias_regularizer)

        self.sigma_bias = self.add_weight(shape=(self.units,),
                                        initializer=self.sigma_initializer,
                                        name='sigma_bias',
                                        constraint=self.bias_constraint,
                                        regularizer=self.bias_regularizer)

        super(NoisyNetDense, self).build(input_shape=input_shape)

    def call(self, x):
        #sample from noise distribution
        e_i = K.random_normal((self.input_dim, self.units))
        e_j = K.random_normal((self.units,))

        #We use the factorized Gaussian noise variant from Section 3 of Fortunato et al.
        eW = K.sign(e_i)*(K.sqrt(K.abs(e_i))) * K.sign(e_j)*(K.sqrt(K.abs(e_j)))
        eB = K.sign(e_j)*(K.abs(e_j)**(1/2))

        #See section 3 of Fortunato et al.
        noise_injected_weights = K.dot(x, self.mu_weight + (self.sigma_weight * eW))
        noise_injected_bias = self.mu_bias + (self.sigma_bias * eB)
        output = K.bias_add(noise_injected_weights, noise_injected_bias)
        if self.activation != None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'mu_initializer': initializers.serialize(self.mu_initializer),
            'sigma_initializer': initializers.serialize(self.sigma_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(NoisyNetDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
