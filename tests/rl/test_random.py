import pytest
import numpy as np
from rl.random import IndependentGaussianProcess
from numpy.testing import assert_allclose

from keras import backend as K

def test_gaussian():
    test_mu = np.array([ [0.0, 1.0], [1.0, 0.0], [-1.0, 2.0] ])
    test_sigma = np.log(np.array([ [2.0, 3.0], [4.0, 5.0], [ 6.0, 0.0] ]))
    test_in = np.array([ [0.0 + 2.0 * 1, 1.0 - 3.0 * 2],
                         [1.0 + 4.0 * 0, 0.0 - 5.0 * 0.5],
                         [-1.0 + 6.0 * 1.5, 2.0] ])
    test_data = (K.constant(test_mu), K.constant(test_sigma), K.constant(test_in))
    p = IndependentGaussianProcess(2)
    result = K.eval(p.get_dist(test_data))
    assert result.shape == (3, 1)
    assert_allclose(result, np.array([ [-6.12964], [-4.95861], [np.nan] ]), equal_nan=True, atol=1e-5)

if __name__ == '__main__':
    pytest.main([__file__])
