import unittest

import chainer
from chainer import testing
import numpy as np

from chainerrl.links import noisy_linear


@testing.parameterize(*testing.product({
    'nobias': [False, True],
}))
class TestFactorizedNoisyLinear(unittest.TestCase):
    def test_calls(self):
        l_uninitialized_new = noisy_linear.FactorizedNoisyLinear(
            5, nobias=self.nobias)
        l_uninitialized_old = noisy_linear.FactorizedNoisyLinear(
            None, 5, nobias=self.nobias)
        l_initialized = noisy_linear.FactorizedNoisyLinear(
            6, 5, nobias=self.nobias)
        for l in [l_uninitialized_new, l_uninitialized_old, l_initialized]:
            x_data = np.arange(12).astype(np.float32).reshape((2, 6))
            x = chainer.Variable(x_data)
            l(x)
            l(x_data + 1)
            l(x_data.reshape((2, 3, 2)))
