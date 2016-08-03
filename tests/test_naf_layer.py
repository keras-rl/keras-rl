# TODO: this needs to be turned into a proper test. Right now this is used to manually test
# if the computations of the NAF layer are correct. This is done by computing a reference value
# in Numpy and comparing it to the output of the NAF layer.
import numpy as np

from keras.models import Model
from keras.layers import Input, merge

from rl.agents.dqn import NAFLayer


nb_actions = 5
batch_size = 32

L_flat_input = Input(shape=((nb_actions * nb_actions + nb_actions) / 2,))
mu_input = Input(shape=(nb_actions,))
action_input = Input(shape=(nb_actions,))
x = merge([L_flat_input, mu_input, action_input], mode='concat')
x = NAFLayer(nb_actions)(x)
model = Model(input=[L_flat_input, mu_input, action_input], output=x)
model.compile(loss='mse', optimizer='sgd')
print(model.summary())


L_flat = np.random.random((batch_size, (nb_actions * nb_actions + nb_actions) / 2)).astype('float32')
mu = np.random.random((batch_size, nb_actions)).astype('float32')
action = np.random.random((batch_size, nb_actions)).astype('float32')

L = np.zeros((batch_size, nb_actions, nb_actions)).astype('float32')
LT = np.copy(L)
for l, l_T, l_flat in zip(L, LT, L_flat):
    l[np.tril_indices(nb_actions)] = l_flat
    l[np.diag_indices(nb_actions)] = np.exp(l[np.diag_indices(nb_actions)])
    l_T[:, :] = l.T
P = np.array([np.dot(l, l_T) for l, l_T in zip(L, LT)]).astype('float32')
A_ref = -.5 * np.array([np.dot(np.dot(a - m, p), a - m) for a, m, p in zip(action, mu, P)]).astype('float32')

A_net = model.predict([L_flat, mu, action]).flatten()
print A_ref, A_net
print np.allclose(A_ref, A_net)
