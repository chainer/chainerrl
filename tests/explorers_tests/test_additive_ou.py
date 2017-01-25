from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
import unittest

import numpy as np

from chainerrl.explorers.additive_ou import AdditiveOU


class TestAdditiveOU(unittest.TestCase):

    def test(self):

        action_size = 3

        def greedy_action_func():
            return np.asarray([0] * action_size, dtype=np.float32)

        explorer = AdditiveOU()

        for t in range(100):
            a = explorer.select_action(t, greedy_action_func)
            print(t, a)
