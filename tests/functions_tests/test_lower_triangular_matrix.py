from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
import unittest

import numpy

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr

from chainerrl.functions.lower_triangular_matrix import lower_triangular_matrix
from chainerrl.functions.lower_triangular_matrix import LowerTriangularMatrix


@testing.parameterize(
    {'n': 1},
    {'n': 2},
    {'n': 3},
    {'n': 4},
    {'n': 5},
)
class TestLowerTriangularMatrix(unittest.TestCase):

    def setUp(self):
        self.batch_size = 5
        self.diag = numpy.random.uniform(
            0.1, 1, (self.batch_size, self.n)).astype(numpy.float32)
        non_diag_size = self.n * (self.n - 1) // 2
        self.non_diag = numpy.random.uniform(
            -1, 1, (self.batch_size, non_diag_size)).astype(numpy.float32)
        self.gy = numpy.random.uniform(
            -1, 1, (self.batch_size, self.n, self.n)).astype(numpy.float32)

    def check_forward(self, diag_data, non_diag_data):
        diag = chainer.Variable(diag_data)
        non_diag = chainer.Variable(non_diag_data)
        y = lower_triangular_matrix(diag, non_diag)

        correct_y = numpy.zeros(
            (self.batch_size, self.n, self.n), dtype=numpy.float32)

        tril_rows, tril_cols = numpy.tril_indices(self.n, -1)
        correct_y[:, tril_rows, tril_cols] = cuda.to_cpu(non_diag_data)

        diag_rows, diag_cols = numpy.diag_indices(self.n)
        correct_y[:, diag_rows, diag_cols] = cuda.to_cpu(diag_data)

        gradient_check.assert_allclose(correct_y, cuda.to_cpu(y.data))

    def test_forward_cpu(self):
        self.check_forward(self.diag, self.non_diag)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.diag), cuda.to_gpu(self.non_diag))

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(
            LowerTriangularMatrix(),
            x_data, y_grad, eps=1e-2, rtol=1e-2)

    def test_backward_cpu(self):
        self.check_backward((self.diag, self.non_diag), self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward((cuda.to_gpu(self.diag), cuda.to_gpu(
            self.non_diag)), cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)
