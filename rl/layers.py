import numpy as np
from keras import initializers, regularizers, activations, constraints
from keras.engine.topology import Layer
import keras.backend as K

class NoisyNetDense(Layer):
    """
    A modified fully-connected layer that injects noise into the parameter distribution
    before each prediction. This randomness forces the agent to explore - at least
    until it can adjust its parameters to learn around it.

    To use: replace Dense layers (like the classifier at the end of a DQN model)
    with NoisyNetDense layers and set your policy to GreedyQ.
    See examples/noisynet_pdd_dqn_atari.py

    Reference: https://arxiv.org/abs/1706.10295
    """
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
        self.bias_constraint = constraints.get(bias_constraint) if kernel_constraint is not None else None
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


class NAFLayer(Layer):
    """Write me
    """
    def __init__(self, nb_actions, mode='full', **kwargs):
        if mode not in ('full', 'diag'):
            raise RuntimeError('Unknown mode "{}" in NAFLayer.'.format(mode))

        self.nb_actions = nb_actions
        self.mode = mode
        super(NAFLayer, self).__init__(**kwargs)

    def call(self, x, mask=None):
        # TODO: validate input shape

        assert (len(x) == 3)
        L_flat = x[0]
        mu = x[1]
        a = x[2]

        if self.mode == 'full':
            # Create L and L^T matrix, which we use to construct the positive-definite matrix P.
            L = None
            LT = None
            if K.backend() == 'theano':
                import theano.tensor as T
                import theano

                def fn(x, L_acc, LT_acc):
                    x_ = K.zeros((self.nb_actions, self.nb_actions))
                    x_ = T.set_subtensor(x_[np.tril_indices(self.nb_actions)], x)
                    diag = K.exp(T.diag(x_)) + K.epsilon()
                    x_ = T.set_subtensor(x_[np.diag_indices(self.nb_actions)], diag)
                    return x_, x_.T

                outputs_info = [
                    K.zeros((self.nb_actions, self.nb_actions)),
                    K.zeros((self.nb_actions, self.nb_actions)),
                ]
                results, _ = theano.scan(fn=fn, sequences=L_flat, outputs_info=outputs_info)
                L, LT = results
            elif K.backend() == 'tensorflow':
                import tensorflow as tf

                # Number of elements in a triangular matrix.
                nb_elems = (self.nb_actions * self.nb_actions + self.nb_actions) // 2

                # Create mask for the diagonal elements in L_flat. This is used to exponentiate
                # only the diagonal elements, which is done before gathering.
                diag_indeces = [0]
                for row in range(1, self.nb_actions):
                    diag_indeces.append(diag_indeces[-1] + (row + 1))
                diag_mask = np.zeros(1 + nb_elems)  # +1 for the leading zero
                diag_mask[np.array(diag_indeces) + 1] = 1
                diag_mask = K.variable(diag_mask)

                # Add leading zero element to each element in the L_flat. We use this zero
                # element when gathering L_flat into a lower triangular matrix L.
                nb_rows = tf.shape(L_flat)[0]
                zeros = tf.expand_dims(tf.tile(K.zeros((1,)), [nb_rows]), 1)
                try:
                    # Old TF behavior.
                    L_flat = tf.concat(1, [zeros, L_flat])
                except TypeError:
                    # New TF behavior
                    L_flat = tf.concat([zeros, L_flat], 1)

                # Create mask that can be used to gather elements from L_flat and put them
                # into a lower triangular matrix.
                tril_mask = np.zeros((self.nb_actions, self.nb_actions), dtype='int32')
                tril_mask[np.tril_indices(self.nb_actions)] = range(1, nb_elems + 1)

                # Finally, process each element of the batch.
                init = [
                    K.zeros((self.nb_actions, self.nb_actions)),
                    K.zeros((self.nb_actions, self.nb_actions)),
                ]

                def fn(a, x):
                    # Exponentiate everything. This is much easier than only exponentiating
                    # the diagonal elements, and, usually, the action space is relatively low.
                    x_ = K.exp(x) + K.epsilon()
                    # Only keep the diagonal elements.
                    x_ *= diag_mask
                    # Add the original, non-diagonal elements.
                    x_ += x * (1. - diag_mask)
                    # Finally, gather everything into a lower triangular matrix.
                    L_ = tf.gather(x_, tril_mask)
                    return [L_, tf.transpose(L_)]

                tmp = tf.scan(fn, L_flat, initializer=init)
                if isinstance(tmp, (list, tuple)):
                    # TensorFlow 0.10 now returns a tuple of tensors.
                    L, LT = tmp
                else:
                    # Old TensorFlow < 0.10 returns a shared tensor.
                    L = tmp[:, 0, :, :]
                    LT = tmp[:, 1, :, :]
            else:
                raise RuntimeError('Unknown Keras backend "{}".'.format(K.backend()))
            assert L is not None
            assert LT is not None
            P = K.batch_dot(L, LT)
        elif self.mode == 'diag':
            if K.backend() == 'theano':
                import theano.tensor as T
                import theano

                def fn(x, P_acc):
                    x_ = K.zeros((self.nb_actions, self.nb_actions))
                    x_ = T.set_subtensor(x_[np.diag_indices(self.nb_actions)], x)
                    return x_

                outputs_info = [
                    K.zeros((self.nb_actions, self.nb_actions)),
                ]
                P, _ = theano.scan(fn=fn, sequences=L_flat, outputs_info=outputs_info)
            elif K.backend() == 'tensorflow':
                import tensorflow as tf

                # Create mask that can be used to gather elements from L_flat and put them
                # into a diagonal matrix.
                diag_mask = np.zeros((self.nb_actions, self.nb_actions), dtype='int32')
                diag_mask[np.diag_indices(self.nb_actions)] = range(1, self.nb_actions + 1)

                # Add leading zero element to each element in the L_flat. We use this zero
                # element when gathering L_flat into a lower triangular matrix L.
                nb_rows = tf.shape(L_flat)[0]
                zeros = tf.expand_dims(tf.tile(K.zeros((1,)), [nb_rows]), 1)
                try:
                    # Old TF behavior.
                    L_flat = tf.concat(1, [zeros, L_flat])
                except TypeError:
                    # New TF behavior
                    L_flat = tf.concat([zeros, L_flat], 1)

                # Finally, process each element of the batch.
                def fn(a, x):
                    x_ = tf.gather(x, diag_mask)
                    return x_

                P = tf.scan(fn, L_flat, initializer=K.zeros((self.nb_actions, self.nb_actions)))
            else:
                raise RuntimeError('Unknown Keras backend "{}".'.format(K.backend()))
        assert P is not None
        assert K.ndim(P) == 3

        # Combine a, mu and P into a scalar (over the batches). What we compute here is
        # -.5 * (a - mu)^T * P * (a - mu), where * denotes the dot-product. Unfortunately
        # TensorFlow handles vector * P slightly suboptimal, hence we convert the vectors to
        # 1xd/dx1 matrices and finally flatten the resulting 1x1 matrix into a scalar. All
        # operations happen over the batch size, which is dimension 0.
        prod = K.batch_dot(K.expand_dims(a - mu, 1), P)
        prod = K.batch_dot(prod, K.expand_dims(a - mu, -1))
        A = -.5 * K.batch_flatten(prod)
        assert K.ndim(A) == 2
        return A

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)

    def compute_output_shape(self, input_shape):
        if len(input_shape) != 3:
            raise RuntimeError("Expects 3 inputs: L, mu, a")
        for i, shape in enumerate(input_shape):
            if len(shape) != 2:
                raise RuntimeError("Input {} has {} dimensions but should have 2".format(i, len(shape)))
        assert self.mode in ('full','diag')
        if self.mode == 'full':
            expected_elements = (self.nb_actions * self.nb_actions + self.nb_actions) // 2
        elif self.mode == 'diag':
            expected_elements = self.nb_actions
        else:
            expected_elements = None
        assert expected_elements is not None
        if input_shape[0][1] != expected_elements:
            raise RuntimeError("Input 0 (L) should have {} elements but has {}".format(expected_elements, input_shape[0][1]))
        if input_shape[1][1] != self.nb_actions:
            raise RuntimeError(
                "Input 1 (mu) should have {} elements but has {}".format(self.nb_actions, input_shape[1][1]))
        if input_shape[2][1] != self.nb_actions:
            raise RuntimeError(
                "Input 2 (action) should have {} elements but has {}".format(self.nb_actions, input_shape[1][1]))
        return input_shape[0][0], 1
