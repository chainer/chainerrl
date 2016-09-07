from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from future import standard_library
from future.utils import with_metaclass
standard_library.install_aliases()

from logging import getLogger

from chainer import cuda
import numpy as np

import explorer


class AdditiveOU(explorer.Explorer):
    """Additive Ornstein-Uhlenbeck process.

    Used in https://arxiv.org/abs/1509.02971."""

    def __init__(self, shape, mu=0.0, theta=0.15, sigma=0.3,
                 dt=1.0, logger=getLogger(__name__)):
        self.shape = shape
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.logger = logger

        self.wiener_state = np.zeros(self.shape, dtype=np.float32)
        self.ou_state = np.full(self.shape, self.mu, dtype=np.float32)

    def evolve(self):
        self.wiener_state += np.random.normal(
            size=self.shape, loc=0, scale=np.sqrt(self.dt))
        self.ou_state += self.theta * \
            (self.mu - self.ou_state) * self.dt + self.sigma * self.wiener_state

    def select_action(self, t, greedy_action_func):
        self.evolve()
        a = greedy_action_func()
        noise = self.ou_state
        self.logger.debug('t:%s noise:%s', t, noise)
        if isinstance(a, cuda.cupy.ndarray):
            noise = cuda.to_gpu(noise)
        return a + noise

    def __repr__(self):
        return 'AdditiveOU(mu={}, theta={}, sigma={})'.format(
            self.mu, self.theta, self.sigma)
