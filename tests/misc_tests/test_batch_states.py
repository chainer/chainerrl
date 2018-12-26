from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()  # NOQA
import unittest

import chainer
from chainer import testing
import numpy as np

import chainerrl


class TestBatchStates(unittest.TestCase):

    def _test(self, xp):

        # state: ((2,2)-shaped array, integer, (1,)-shaped array)
        states = [
            (np.arange(4).reshape((2, 2)), 0, np.zeros(1)),
            (np.arange(4).reshape((2, 2)) + 1, 1, np.zeros(1) + 1),
        ]

        def phi(state):
            return state[0] * 2, state[1], state[2] * 3

        batch = chainerrl.misc.batch_states(states, xp=xp, phi=phi)
        self.assertIsInstance(batch, tuple)
        batch_a, batch_b, batch_c = batch
        xp.testing.assert_allclose(
            batch_a,
            xp.asarray([
                [[0, 2],
                 [4, 6]],
                [[2, 4],
                 [6, 8]],
            ])
        )
        xp.testing.assert_allclose(
            batch_b,
            xp.asarray([0, 1])
        )
        xp.testing.assert_allclose(
            batch_c,
            xp.asarray([
                [0],
                [3],
            ])
        )

    def test_cpu(self):
        self._test(np)

    @testing.attr.gpu
    def test_gpu(self):
        self._test(chainer.cuda.cupy)
