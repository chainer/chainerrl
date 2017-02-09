from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

import unittest

from chainerrl.links import Sequence


class TestSequence(unittest.TestCase):

    def test_call(self):

        def func_a(x):
            return x + 1

        b_test_mode = [False]

        def func_b(x, test=False):
            b_test_mode[0] = test
            return x + 1

        c_test_mode = [False]
        c_hoge_mode = [False]

        def func_c(x, test=False, hoge=False):
            c_test_mode[0] = test
            c_hoge_mode[0] = hoge
            return x + 1

        def _test_call(seq):

            out = seq(1)
            self.assertEqual(out, 4)
            self.assertFalse(b_test_mode[0])
            self.assertFalse(c_test_mode[0])
            self.assertFalse(c_hoge_mode[0])

            out = seq(1, test=True)
            self.assertEqual(out, 4)
            self.assertTrue(b_test_mode[0])
            self.assertTrue(c_test_mode[0])
            self.assertFalse(c_hoge_mode[0])

            out = seq(1, test=True, hoge=True)
            self.assertEqual(out, 4)
            self.assertTrue(b_test_mode[0])
            self.assertTrue(c_test_mode[0])
            self.assertTrue(c_hoge_mode[0])

            out = seq(1, test=False, hoge=True)
            self.assertEqual(out, 4)
            self.assertFalse(b_test_mode[0])
            self.assertFalse(c_test_mode[0])
            self.assertTrue(c_hoge_mode[0])

        _test_call(Sequence(func_a, func_b, func_c))
        _test_call(Sequence(Sequence(func_a, func_b, func_c)))
        _test_call(Sequence(Sequence(func_a),
                            Sequence(func_b), Sequence(func_c)))
