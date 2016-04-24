import unittest
import random

import numpy as np
import chainer
from chainer import links as L

import copy_param


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
