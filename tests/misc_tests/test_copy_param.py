from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()  # NOQA
import unittest

import chainer
from chainer import links as L
import numpy as np

from chainerrl.misc import copy_param


class TestCopyParam(unittest.TestCase):

    def test_copy_param(self):
        a = L.Linear(1, 5)
        b = L.Linear(1, 5)

        s = chainer.Variable(np.random.rand(1, 1).astype(np.float32))
        a_out = list(a(s).array.ravel())
        b_out = list(b(s).array.ravel())
        self.assertNotEqual(a_out, b_out)

        # Copy b's parameters to a
        copy_param.copy_param(a, b)

        a_out_new = list(a(s).array.ravel())
        b_out_new = list(b(s).array.ravel())
        self.assertEqual(a_out_new, b_out)
        self.assertEqual(b_out_new, b_out)

    def test_copy_param_scalar(self):
        a = chainer.Chain()
        with a.init_scope():
            a.p = chainer.Parameter(np.array(1))
        b = chainer.Chain()
        with b.init_scope():
            b.p = chainer.Parameter(np.array(2))

        self.assertNotEqual(a.p.array, b.p.array)

        # Copy b's parameters to a
        copy_param.copy_param(a, b)

        self.assertEqual(a.p.array, b.p.array)

    def test_copy_param_type_check(self):
        a = L.Linear(None, 5)
        b = L.Linear(1, 5)

        with self.assertRaises(TypeError):
            # Copy b's parameters to a, but since `a` parameter is not
            # initialized, it should raise error.
            copy_param.copy_param(a, b)

    def test_soft_copy_param(self):
        a = L.Linear(1, 5)
        b = L.Linear(1, 5)

        a.W.array[:] = 0.5
        b.W.array[:] = 1

        # a = (1 - tau) * a + tau * b
        copy_param.soft_copy_param(target_link=a, source_link=b, tau=0.1)

        np.testing.assert_almost_equal(a.W.array, np.full(a.W.shape, 0.55))
        np.testing.assert_almost_equal(b.W.array, np.full(b.W.shape, 1.0))

        copy_param.soft_copy_param(target_link=a, source_link=b, tau=0.1)

        np.testing.assert_almost_equal(a.W.array, np.full(a.W.shape, 0.595))
        np.testing.assert_almost_equal(b.W.array, np.full(b.W.shape, 1.0))

    def test_soft_copy_param_scalar(self):
        a = chainer.Chain()
        with a.init_scope():
            a.p = chainer.Parameter(np.array(0.5))
        b = chainer.Chain()
        with b.init_scope():
            b.p = chainer.Parameter(np.array(1))

        # a = (1 - tau) * a + tau * b
        copy_param.soft_copy_param(target_link=a, source_link=b, tau=0.1)

        np.testing.assert_almost_equal(a.p.array, 0.55)
        np.testing.assert_almost_equal(b.p.array, 1.0)

        copy_param.soft_copy_param(target_link=a, source_link=b, tau=0.1)

        np.testing.assert_almost_equal(a.p.array, 0.595)
        np.testing.assert_almost_equal(b.p.array, 1.0)

    def test_soft_copy_param_type_check(self):
        a = L.Linear(None, 5)
        b = L.Linear(1, 5)

        with self.assertRaises(TypeError):
            copy_param.soft_copy_param(target_link=a, source_link=b, tau=0.1)
