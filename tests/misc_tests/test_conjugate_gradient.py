from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
import unittest

import chainer
from chainer import testing
from chainer.testing import condition
import numpy as np

import chainerrl


def inv_mat(mat):
    # Always use numpy.linalg.inv because cupy v1 doesn't support it
    xp = chainer.cuda.get_array_module(mat)
    if xp is np:
        return np.linalg.inv(mat)
    else:
        return chainer.cuda.to_gpu(
            np.linalg.inv(chainer.cuda.to_cpu(mat)))


@testing.parameterize(
    *testing.product({
        'n': [1, 5],
    })
)
class TestConjugateGradient(unittest.TestCase):

    def _test(self, xp):
        # A must be symmetric and positive-definite
        random_mat = xp.random.normal(size=(self.n, self.n))
        A = random_mat.dot(random_mat.T)
        inv_A = inv_mat(A)
        b = xp.random.normal(size=(self.n,))

        def A_product_func(vec):
            self.assertEqual(xp, chainer.cuda.get_array_module(vec))
            self.assertEqual(vec.shape, b.shape)
            return A.dot(vec)

        x = chainerrl.misc.conjugate_gradient(A_product_func, b)
        self.assertTrue(chainer.cuda.get_array_module(x), xp)
        xp.testing.assert_allclose(x, inv_A.dot(b))

    @condition.retry(3)
    def test_cpu(self):
        self._test(np)

    @testing.attr.gpu
    @condition.retry(3)
    def test_gpu(self):
        self._test(chainer.cuda.cupy)
