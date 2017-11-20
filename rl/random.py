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

# For PPO use
# sigma is a np array of shape (n,)
class IndependentGaussianProcess(object):
    def __init__(self, sigma):
        dim = sigma.shape
        assert len(dim) == 1
        self.sigma = sigma
        self.n = dim[0]

    def get_param(self):
        return self.sigma

    def set_param(self, sigma):
        dim = sigma.shape
        assert len(dim) == 1 and dim[0] == self.n
        self.sigma = sigma

    # generate a random sample according to current distribution (specified by param sigma)
    # Input: mu: np array of shape (n,)
    # Output: np array of shape (n,)
    def sample(self, mu):
        assert mu.shape == (self.n,)
        return np.random.normal(mu, self.sigma)

    # return a keras layer to compute the log likelihood of selected action given current state
    # Output: a lambda layer that accepts three arguments: [action, actor_out, sigma]
    # and return a keras tensor of dimension (batch, 1)
    def get_dist(self):
        return Lambda(
            lambda x:
                K.constant(- self.n * math.log(2*math.pi) / 2 ) - K.sum(x[2], axis=1, keepdims=True)
                - K.sum(K.exp( 2 * K.log(K.abs(x[0] - x[1])) - K.constant(math.log(2.0)) - 2 * x[2]), axis=1, keepdims=True))
