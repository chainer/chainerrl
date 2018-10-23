from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()  # NOQA
import unittest

import numpy

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr

import chainerrl


@testing.parameterize(
    *testing.product({
        'batchsize': [1, 3],
        'n': [1, 2, 7],
        'shape': [(1,), (1, 1), (2,), (2, 3)],
    })
)
class TestSumArrays(unittest.TestCase):

    def setUp(self):
        self.batch_size = 5
        array_shape = (self.batchsize,) + self.shape
        self.xs = [numpy.random.uniform(
            -1, 1, array_shape).astype(numpy.float32)
            for _ in range(self.n)]
        self.gy = numpy.random.uniform(
            -1, 1, array_shape).astype(numpy.float32)

    def check_forward(self, xs):
        y = chainerrl.functions.sum_arrays(xs)
        correct_y = sum(self.xs)
        gradient_check.assert_allclose(correct_y, cuda.to_cpu(y.array))

    def test_forward_cpu(self):
        self.check_forward(self.xs)

    @attr.gpu
    def test_forward_gpu(self):
        xs_gpu = [chainer.cuda.to_gpu(x) for x in self.xs]
        self.check_forward(xs_gpu)

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(
            chainerrl.functions.SumArrays(),
            x_data, y_grad, eps=1e-2, rtol=1e-2)

    def test_backward_cpu(self):
        self.check_backward(self.xs, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        xs_gpu = [chainer.cuda.to_gpu(x) for x in self.xs]
        self.check_backward(xs_gpu, cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)
