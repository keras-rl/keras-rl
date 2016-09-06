from __future__ import division
import pytest
import numpy as np
from numpy.testing import assert_allclose

from keras.models import Model, Sequential
from keras.layers import Input, Dense, merge
from keras.optimizers import SGD
import keras.backend as K

from rl.util import clone_optimizer, clone_model


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
    x = Dense(8)(merge([in1, in2], mode='concat'))
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


if __name__ == '__main__':
    pytest.main([__file__])
