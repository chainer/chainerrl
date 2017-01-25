from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import unittest

from chainer import cuda
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
import numpy

from chainerrl.functions.mellowmax import maximum_entropy_mellowmax
from chainerrl.functions.mellowmax import mellowmax


@testing.parameterize(*testing.product({
    'shape': [(1, 1), (2, 3), (2, 3, 4), (2, 3, 4, 5)],
    'dtype': [numpy.float32],
    'omega': [10, 5, 1, -1, -5, -10],
    'axis': [0, 1, -1, -2],
    'same_value': [True, False],
}))
class TestMellowmax(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        if self.same_value:
            self.x[:] = numpy.random.uniform(-1, 1, 1).astype(self.dtype)

    def check_forward(self, x_data):
        xp = cuda.get_array_module(x_data)
        y = mellowmax(x_data, axis=self.axis, omega=self.omega)
        self.assertEqual(y.data.dtype, self.dtype)

        x_min = xp.min(x_data, axis=self.axis)
        x_max = xp.max(x_data, axis=self.axis)
        x_mean = xp.mean(x_data, axis=self.axis)
        print('x_min', x_min)
        print('y.data', y.data)

        # min <= mellowmax <= max
        eps = 1e-5
        self.assertTrue(xp.all(x_min <= y.data + eps))
        self.assertTrue(xp.all(x_max >= y.data - eps))

        # omega > 0 -> mellowmax is more like max
        if self.omega > 0:
            self.assertTrue(xp.all(x_mean <= y.data + eps))
        # omega < 0 -> mellowmax is more like min
        if self.omega < 0:
            self.assertTrue(xp.all(x_mean >= y.data - eps))

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))


@testing.parameterize(*testing.product({
    'shape': [(1, 1), (2, 3), (2, 3, 4), (2, 3, 4, 5)],
    'dtype': [numpy.float32],
    'omega': [10, 5, 1, 0, -1, -5, -10],
    'same_value': [True, False],
}))
class TestMaximumEntropyMellowmax(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        if self.same_value:
            self.x[:] = numpy.random.uniform(-1, 1, 1).astype(self.dtype)

    def check_forward(self, x_data):
        xp = cuda.get_array_module(x_data)
        y = maximum_entropy_mellowmax(x_data)
        self.assertEqual(y.data.dtype, self.dtype)

        print('y', y.data)

        # Outputs must be positive
        xp.testing.assert_array_less(xp.zeros_like(y.data), y.data)

        # Sums must be ones
        sums = xp.sum(y.data, axis=1)
        testing.assert_allclose(sums, xp.ones_like(sums))

        # Expectations must be equal to memllowmax's outputs
        testing.assert_allclose(
            xp.sum(y.data * x_data, axis=1), mellowmax(x_data, axis=1).data)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

testing.run_module(__name__, __file__)
