from __future__ import division
import pytest
import numpy as np
from numpy.testing import assert_allclose

from keras.models import Model
from keras.layers import Input, merge

from rl.agents.dqn import NAFLayer


def test_naf_layer():
    batch_size = 2
    for nb_actions in (1, 3):
        # Construct single model with NAF as the only layer, hence it is fully deterministic
        # since no weights are used, which would be randomly initialized.
        L_flat_input = Input(shape=((nb_actions * nb_actions + nb_actions) // 2,))
        mu_input = Input(shape=(nb_actions,))
        action_input = Input(shape=(nb_actions,))
        x = merge([L_flat_input, mu_input, action_input], mode='concat')
        x = NAFLayer(nb_actions)(x)
        model = Model(input=[L_flat_input, mu_input, action_input], output=x)
        model.compile(loss='mse', optimizer='sgd')
        
        # Create random test data.
        L_flat = np.random.random((batch_size, (nb_actions * nb_actions + nb_actions) // 2)).astype('float32')
        mu = np.random.random((batch_size, nb_actions)).astype('float32')
        action = np.random.random((batch_size, nb_actions)).astype('float32')

        # Perform reference computations in numpy since these are much easier to verify.
        L = np.zeros((batch_size, nb_actions, nb_actions)).astype('float32')
        LT = np.copy(L)
        for l, l_T, l_flat in zip(L, LT, L_flat):
            l[np.tril_indices(nb_actions)] = l_flat
            l[np.diag_indices(nb_actions)] = np.exp(l[np.diag_indices(nb_actions)])
            l_T[:, :] = l.T
        P = np.array([np.dot(l, l_T) for l, l_T in zip(L, LT)]).astype('float32')
        A_ref = np.array([np.dot(np.dot(a - m, p), a - m) for a, m, p in zip(action, mu, P)]).astype('float32')
        A_ref *= -.5

        # Finally, compute the output of the net, which should be identical to the previously
        # computed reference.
        A_net = model.predict([L_flat, mu, action]).flatten()
        assert_allclose(A_net, A_ref, rtol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__])
