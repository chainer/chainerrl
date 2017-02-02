from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
import unittest

from chainer import testing
import numpy as np

from chainerrl.explorers.additive_ou import AdditiveOU


@testing.parameterize(*testing.product({
    'action_size': [1, 3],
    'sigma_type': ['scalar', 'ndarray'],
}))
class TestAdditiveOU(unittest.TestCase):

    def test(self):

        def greedy_action_func():
            return np.asarray([0] * self.action_size, dtype=np.float32)

        if self.sigma_type == 'scalar':
            sigma = np.random.rand()
        elif self.sigma_type == 'ndarray':
            sigma = np.random.rand(self.action_size)
        theta = np.random.rand()

        explorer = AdditiveOU(theta=theta, sigma=sigma)

        print('theta:', theta, 'sigma', sigma)
        for t in range(100):
            a = explorer.select_action(t, greedy_action_func)
            print(t, a)
