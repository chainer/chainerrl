from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
import unittest

import numpy as np

from explorers.additive_ou import AdditiveOU


class TestAdditiveOU(unittest.TestCase):

    def test(self):

        action_size = 3
        dt = 0.5
        sigma = 0.001
        theta = 0.3

        def greedy_action_func():
            return np.asarray([0] * action_size, dtype=np.float32)

        explorer = AdditiveOU(action_size, dt=dt, theta=theta, sigma=sigma)

        for t in range(10000):
            a = explorer.select_action(t, greedy_action_func)
            print(a)
