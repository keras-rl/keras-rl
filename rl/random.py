from __future__ import division
import numpy as np

from keras import backend as K
from keras.layers import Lambda
import math

class RandomProcess(object):
    def reset_states(self):
        pass


class AnnealedGaussianProcess(RandomProcess):
    def __init__(self, mu, sigma, sigma_min, n_steps_annealing):
        self.mu = mu
        self.sigma = sigma
        self.n_steps = 0

        if sigma_min is not None:
            self.m = -float(sigma - sigma_min) / float(n_steps_annealing)
            self.c = sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0.
            self.c = sigma
            self.sigma_min = sigma

    @property
    def current_sigma(self):
        sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)
        return sigma


class GaussianWhiteNoiseProcess(AnnealedGaussianProcess):
    def __init__(self, mu=0., sigma=1., sigma_min=None, n_steps_annealing=1000, size=1):
        super(GaussianWhiteNoiseProcess, self).__init__(mu=mu, sigma=sigma, sigma_min=sigma_min, n_steps_annealing=n_steps_annealing)
        self.size = size

    def sample(self):
        sample = np.random.normal(self.mu, self.current_sigma, self.size)
        self.n_steps += 1
        return sample

# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckProcess(AnnealedGaussianProcess):
    def __init__(self, theta, mu=0., sigma=1., dt=1e-2, x0=None, size=1, sigma_min=None, n_steps_annealing=1000):
        super(OrnsteinUhlenbeckProcess, self).__init__(mu=mu, sigma=sigma, sigma_min=sigma_min, n_steps_annealing=n_steps_annealing)
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset_states()

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.current_sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.x_prev = x
        self.n_steps += 1
        return x

    def reset_states(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)

class ProbabilityDistribution(object):
    """Specify a parametric family of fixed probability distribution. Currently used by the ``PPOAgent`` class.
    Implementation of this interface should supports both methods ``sample`` and ``get_dist``."""
    def sample(self, x):
        """
        Generate a single random sample according to the probability distribution.

        :param x: Python list of parameters for the random vector, where each parameter is a numpy array.
        :return: Sampled value. Should be a numpy array with dimension same as ``self.sample_dim()``.
        """
        pass

    def sample_dim(self):
        """
        Get the tensor dimension spec for the generated random sample
        :return: Dimension of the random samples
        """
        pass

    def get_dist(self, x):
        """
        Return a keras expression to compute the log likelihood of selected value given a fixed parameter.

        :param x: Python list of keras function, consisting of all parameters for the random vector (in the same format as in ``sample``), plus the sampled value at the end. All values come in batched form.
        :return: Keras tensor of dimension ``(batch, 1)``

        .. note::
        In application, the actor network should output exactly the parameters for the random vector, which is then
        passed to ``sample`` to generate the actual action.
        """
        pass


class IndependentGaussianProcess(ProbabilityDistribution):
    """
    Specifies a Multivariate Gaussian/Normal Distribution, with a diagonal covariance matrix (i.e. each component
    in the random vector are independent).

    The random parameters are ``mu`` (mean) and ``log_sigma`` (natural log of standard deviation), both numpy
    array of shape ``(n,)``, where n is the dimension of the random vector.

    The generated output is also a numpy array of shape ``(n,)``.

    .. seealso:: :class:`ProbabilityDistribution`
    """
    def __init__(self, n):
        self.n = n

    def sample(self, x):
        mu, log_sigma = x
        assert mu.shape == (self.n,)
        assert log_sigma.shape == (self.n,)
        return np.random.normal(mu, np.exp(log_sigma))

    def sample_dim(self):
        return (self.n,)

    def get_dist(self, x):
        mu, log_sigma, y = x
        return K.constant(- self.n * math.log(2*math.pi) / 2 ) - K.sum(log_sigma, axis=1, keepdims=True) - \
               K.sum(K.exp( 2 * K.log(K.abs(y - mu)) - K.constant(math.log(2.0)) - 2 * log_sigma), axis=1, keepdims=True)
