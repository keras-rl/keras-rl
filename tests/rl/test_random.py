import pytest
import numpy as np
from rl.random import IndependentGaussianProcess

from keras import backend as K

def test_gaussian():
    test_mu = np.array([ [0.0, 1.0], [1.0, 0.0], [-1.0, 2.0] ])
    test_sigma = np.array([ [2.0, 3.0], [4.0, 5.0], [ 6.0, 0.0] ])
    test_in = np.array([ [0.0 + 2.0 * 1, 1.0 - 3.0 * 2],
                         [1.0 + 4.0 * 0, 0.0 - 5.0 * 0.5],
                         [-1.0 + 6.0 * 1.5, 2.0] ])
    test_data = (K.constant(test_mu), K.constant(test_sigma), K.constant(test_in))
    p = IndependentGaussianProcess(2)
    result = p.get_dist(test_data)
    assert False, K.eval(result)

if __name__ == '__main__':
    pytest.main([__file__])
