import unittest

import chainer
from chainer import cuda
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
import numpy

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
        self.linear = noisy_linear.FactorizedNoisyLinear(mu)

    def _test_calls(self, xp):
        x_data = xp.arange(12).astype(numpy.float32).reshape((2, 6))
        x = chainer.Variable(x_data)
        self.linear(x)
        self.linear(x_data + 1)
        self.linear(x_data.reshape((2, 3, 2)))

    def test_calls_cpu(self):
        self._test_calls(numpy)

    @attr.gpu
    def test_calls_gpu(self):
        self.linear.to_gpu(0)
        self._test_calls(cuda.cupy)

    @attr.gpu
    def test_calls_gpu_after_to_gpu(self):
        mu = self.linear.mu
        mu.to_gpu(0)
        self.linear = noisy_linear.FactorizedNoisyLinear(mu)
        self._test_calls(cuda.cupy)

    def _test_randomness(self, xp):
        x = xp.random.standard_normal((10, 6)).astype(numpy.float32)
        y1 = self.linear(x).array
        y2 = self.linear(x).array
        d = float(xp.mean(xp.square(y1 - y2)))

        # The parameter name suggests that
        # xp.sqrt(d / 2) is approx to sigma_scale = 0.4
        # In fact, (for each element _[i, j],) it holds:
        # \E[(y2 - y1) ** 2] = 2 * \Var(y) = (4 / pi) * sigma_scale ** 2

        target = (0.4 ** 2) * 2
        if self.nobias:
            target *= 2 / numpy.pi
        else:
            target *= 2 / numpy.pi + numpy.sqrt(2 / numpy.pi) / y1.shape[1]

        self.assertGreater(d, target / 3.)
        self.assertLess(d, target * 3.)

    @condition.retry(3)
    def test_randomness_cpu(self):
        self._test_randomness(numpy)

    @attr.gpu
    @condition.retry(3)
    def test_randomness_gpu(self):
        self.linear.to_gpu(0)
        self._test_randomness(cuda.cupy)

    def _test_non_randomness(self, xp):
        # Noises should be the same in a batch
        x0 = xp.random.standard_normal((1, 6)).astype(numpy.float32)
        x = xp.broadcast_to(x0, (2, 6))
        y = self.linear(x).array
        xp.testing.assert_allclose(y[0], y[1], rtol=1e-4)

    def test_non_randomness_cpu(self):
        self._test_non_randomness(numpy)

    @attr.gpu
    def test_non_randomness_gpu(self):
        self.linear.to_gpu(0)
        self._test_non_randomness(cuda.cupy)
