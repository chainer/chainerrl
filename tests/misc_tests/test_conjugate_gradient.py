from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()  # NOQA
import unittest

import chainer
from chainer import testing
from chainer.testing import condition
import numpy as np

import chainerrl


@testing.parameterize(
    *testing.product({
        'n': [1, 5],
        'dtype': [np.float64, np.float32],
    })
)
class TestConjugateGradient(unittest.TestCase):

    def _test(self, xp):
        # A must be symmetric and positive-definite
        random_mat = xp.random.normal(size=(self.n, self.n)).astype(self.dtype)
        A = random_mat.dot(random_mat.T)
        x_ans = xp.random.normal(size=self.n).astype(self.dtype)
        b = A.dot(x_ans)

        def A_product_func(vec):
            self.assertEqual(xp, chainer.cuda.get_array_module(vec))
            self.assertEqual(vec.shape, b.shape)
            return A.dot(vec)

        x = chainerrl.misc.conjugate_gradient(A_product_func, b)
        self.assertEqual(x.dtype, self.dtype)
        self.assertTrue(chainer.cuda.get_array_module(x), xp)
        xp.testing.assert_allclose(x, x_ans, rtol=1e-3)

    @condition.retry(3)
    def test_cpu(self):
        self._test(np)

    @testing.attr.gpu
    @condition.retry(3)
    def test_gpu(self):
        self._test(chainer.cuda.cupy)
