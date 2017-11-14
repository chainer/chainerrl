import unittest

import chainer
from chainer import testing
import numpy as np

from chainerrl.links import noisy_linear


@testing.parameterize(*testing.product({
    'size_args': [
        (5,),  # uninitialized from Chainer v2
        (None, 5),  # uninitialized
        (6, 5),  # initialized
    ],
    'nobias': [False, True],
}))
class TestFactorizedNoisyLinear(unittest.TestCase):
    def setUp(self):
        self.l = noisy_linear.FactorizedNoisyLinear(
            *self.size_args, nobias=self.nobias)

    def test_calls(self):
        x_data = np.arange(12).astype(np.float32).reshape((2, 6))
        x = chainer.Variable(x_data)
        self.l(x)
        self.l(x_data + 1)
        self.l(x_data.reshape((2, 3, 2)))

    @testing.condition.retry(3)
    def test_randomness(self):
        x = np.random.standard_normal((10, 6)).astype(np.float32)
        y1 = self.l(x).data
        y2 = self.l(x).data
        d = np.mean(np.sqrt(np.mean(np.square(y1 - y2), axis=1)))

        # d should be approx to sigma_scale = 0.4.
        # Note: This approximation is not exact.
        target = 0.4
        if not self.nobias:
            target *= np.sqrt(2)

        self.assertGreater(d, target / 3.)
        self.assertLess(d, target * 3.)
