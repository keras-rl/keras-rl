from __future__ import division
import pytest
import numpy as np
from numpy.testing import assert_allclose

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Concatenate
from keras.optimizers.optimizer_experimental.sgd import SGD
import keras.backend as K

from rl.util import clone_optimizer, clone_model, huber_loss, WhiteningNormalizer


def test_clone_sequential_model():
    seq = Sequential()
    seq.add(Dense(8, input_shape=(3,)))
    seq.compile(optimizer='sgd', loss='mse')

    clone = clone_model(seq)
    clone.compile(optimizer='sgd', loss='mse')

    ins = np.random.random((4, 3))
    y_pred_seq = seq.predict_on_batch(ins)
    y_pred_clone = clone.predict_on_batch(ins)
    assert y_pred_seq.shape == y_pred_clone.shape
    assert_allclose(y_pred_seq, y_pred_clone)


def test_clone_graph_model():
    in1 = Input(shape=(2,))
    in2 = Input(shape=(3,))
    x = Dense(8)(Concatenate()([in1, in2]))
    graph = Model([in1, in2], x)
    graph.compile(optimizer='sgd', loss='mse')

    clone = clone_model(graph)
    clone.compile(optimizer='sgd', loss='mse')

    ins = [np.random.random((4, 2)), np.random.random((4, 3))]
    y_pred_graph = graph.predict_on_batch(ins)
    y_pred_clone = clone.predict_on_batch(ins)
    assert y_pred_graph.shape == y_pred_clone.shape
    assert_allclose(y_pred_graph, y_pred_clone)


def test_clone_optimizer():
    lr, momentum, clipnorm, clipvalue = np.random.random(size=4)
    optimizer = SGD(lr=lr, momentum=momentum, clipnorm=clipnorm, clipvalue=clipvalue)
    clone = clone_optimizer(optimizer)

    assert isinstance(clone, SGD)
    assert K.get_value(optimizer.lr) == K.get_value(clone.lr)
    assert K.get_value(optimizer.momentum) == K.get_value(clone.momentum)
    assert optimizer.clipnorm == clone.clipnorm
    assert optimizer.clipvalue == clone.clipvalue


def test_clone_optimizer_from_string():
    clone = clone_optimizer('sgd')
    assert isinstance(clone, SGD)


def test_huber_loss():
    a = np.array([1.,  1.5, 2., 4.])
    b = np.array([1.5, 1.,  4., 2.])
    assert_allclose(K.eval(huber_loss(a, b, 1.)), np.array([.125, .125, 1.5, 1.5]))
    assert_allclose(K.eval(huber_loss(a, b, 3.)), np.array([.125, .125, 2., 2.]))
    assert_allclose(K.eval(huber_loss(a, b, np.inf)), np.array([.125, .125, 2., 2.]))


def test_whitening_normalizer():
    x = np.random.normal(loc=.2, scale=2., size=(1000, 5))
    normalizer = WhiteningNormalizer(shape=(5,))
    normalizer.update(x[:500])
    normalizer.update(x[500:])

    assert_allclose(normalizer.mean, np.mean(x, axis=0))
    assert_allclose(normalizer.std, np.std(x, axis=0))
    
    x_norm = normalizer.normalize(x)
    assert_allclose(np.mean(x_norm, axis=0), np.zeros(5, dtype=normalizer.dtype), atol=1e-5)
    assert_allclose(np.std(x_norm, axis=0), np.ones(5, dtype=normalizer.dtype), atol=1e-5)

    x_denorm = normalizer.denormalize(x_norm)
    assert_allclose(x_denorm, x)


if __name__ == '__main__':
    pytest.main([__file__])
