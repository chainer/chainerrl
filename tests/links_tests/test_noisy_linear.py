import unittest

import chainer
from chainer import testing
from chainer.testing import condition
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
        mu = chainer.links.Linear(*self.size_args, nobias=self.nobias)
        self.l = noisy_linear.FactorizedNoisyLinear(mu)

    def test_calls(self):
        x_data = np.arange(12).astype(np.float32).reshape((2, 6))
        x = chainer.Variable(x_data)
        self.l(x)
        self.l(x_data + 1)
        self.l(x_data.reshape((2, 3, 2)))

    @condition.retry(3)
    def test_randomness(self):
        x = np.random.standard_normal((10, 6)).astype(np.float32)
        y1 = self.l(x).data
        y2 = self.l(x).data
        d = np.mean(np.square(y1 - y2))

        # The parameter name suggests that
        # np.sqrt(d / 2) is approx to sigma_scale = 0.4
        # In fact, (for each element _[i, j],) it holds:
        # \E[(y2 - y1) ** 2] = 2 * \Var(y) = (4 / pi) * sigma_scale ** 2

        target = (0.4 ** 2) * 2
        if self.nobias:
            target *= 2 / np.pi
        else:
            target *= 2 / np.pi + np.sqrt(2 / np.pi)

        self.assertGreater(d, target / 3.)
        self.assertLess(d, target * 3.)

    def test_non_randomness(self):
        # Noises should be the same in a batch
        x0 = np.random.standard_normal((1, 6)).astype(np.float32)
        x = np.broadcast_to(x0, (2, 6))
        y = self.l(x).data
        np.testing.assert_allclose(y[0], y[1], rtol=1e-4)
