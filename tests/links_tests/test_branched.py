from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import unittest

import chainer
from chainer import functions as F
from chainer import links as L
from chainer import testing
import numpy as np

from chainerrl.links import Branched


@testing.parameterize(*(
    testing.product({
        'batch_size': [1, 2],
    })
))
class TestBranched(unittest.TestCase):

    def test_manual(self):
        link1 = L.Linear(2, 3)
        link2 = L.Linear(2, 5)
        link3 = chainer.Sequential(
            L.Linear(2, 7),
            F.tanh,
        )
        plink = Branched(link1, link2, link3)
        x = np.zeros((self.batch_size, 2), dtype=np.float32)
        pout = plink(x)
        self.assertIsInstance(pout, tuple)
        self.assertEqual(len(pout), 3)
        out1 = link1(x)
        out2 = link2(x)
        out3 = link3(x)
        np.testing.assert_allclose(pout[0].array, out1.array)
        np.testing.assert_allclose(pout[1].array, out2.array)
        np.testing.assert_allclose(pout[2].array, out3.array)
