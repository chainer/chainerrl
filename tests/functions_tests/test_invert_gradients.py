from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import unittest

import chainer
from chainer import cuda
from chainer import functions
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
import numpy

from chainerrl.functions.invert_gradients import invert_gradients


@testing.parameterize(*testing.product({
    'shape': [(1, 1), (2, 3), (2, 3, 4), (2, 3, 4, 5)],
    'dtype': [numpy.float32],
}))
class TestInvertGradients(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)

    def check_forward(self, x_data):

        # In chainer, update will be like x.data -= lr * x.grad,
        # which means negative gradients will increase values.

        # Not exceeding
        range_max = x_data + 0.1
        range_min = x_data - 0.1
        x = chainer.Variable(x_data)
        y = invert_gradients(x, range_min=range_min, range_max=range_max)

        loss = functions.sum(y)  # Minimize y
        loss.backward()
        self.assertTrue((x.grad > 0).all())  # Decrease x
        x.cleargrad()

        loss = -functions.sum(y)  # Maximize y
        loss.backward()
        self.assertTrue((x.grad < 0).all())  # Increase x
        x.cleargrad()

        # Exceeding range_max
        range_max = x_data - 0.1
        range_min = x_data - 0.2
        y = invert_gradients(x, range_min=range_min, range_max=range_max)

        loss = functions.sum(y)  # Minimize y
        loss.backward()
        self.assertTrue((x.grad > 0).all())  # Decrease x
        x.cleargrad()

        loss = -functions.sum(y)  # Maximize y
        loss.backward()
        self.assertTrue((x.grad > 0).all())  # Decrease x
        x.cleargrad()

        # Exceeding range_min
        range_max = x_data + 0.2
        range_min = x_data + 0.1
        y = invert_gradients(x, range_min=range_min, range_max=range_max)

        loss = functions.sum(y)  # Minimize y
        loss.backward()
        self.assertTrue((x.grad < 0).all())  # Increase x
        x.cleargrad()

        loss = -functions.sum(y)  # Maximize y
        loss.backward()
        self.assertTrue((x.grad < 0).all())  # Increase x
        x.cleargrad()

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))


testing.run_module(__name__, __file__)
