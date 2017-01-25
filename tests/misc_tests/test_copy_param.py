from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
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
        a_out = list(a(s).data.ravel())
        b_out = list(b(s).data.ravel())
        self.assertNotEqual(a_out, b_out)

        # Copy b's parameters to a
        copy_param.copy_param(a, b)

        a_out_new = list(a(s).data.ravel())
        b_out_new = list(b(s).data.ravel())
        self.assertEqual(a_out_new, b_out)
        self.assertEqual(b_out_new, b_out)

    def test_soft_copy_param(self):
        a = L.Linear(1, 5)
        b = L.Linear(1, 5)

        a.W.data[:] = 0.5
        b.W.data[:] = 1

        # a = (1 - tau) * a + tau * b
        copy_param.soft_copy_param(target_link=a, source_link=b, tau=0.1)

        np.testing.assert_almost_equal(a.W.data, np.full(a.W.data.shape, 0.55))
        np.testing.assert_almost_equal(b.W.data, np.full(b.W.data.shape, 1.0))

        copy_param.soft_copy_param(target_link=a, source_link=b, tau=0.1)

        np.testing.assert_almost_equal(
            a.W.data, np.full(a.W.data.shape, 0.595))
        np.testing.assert_almost_equal(b.W.data, np.full(b.W.data.shape, 1.0))
