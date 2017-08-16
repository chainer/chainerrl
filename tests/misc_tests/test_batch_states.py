from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
import unittest

import chainer
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition

import numpy as np

from chainerrl.misc.batch_states import batch_states


@testing.parameterize(*testing.product({
    'n_states': [1, 10],
    'obs_shape': [(1,),  (3,), (3, 2)],
}))
class TestBatchStates(unittest.TestCase):

    def _test_batch_states(self, input_xp, output_xp):
        observations = [input_xp.ones(self.obs_shape)
                        for _ in range(self.n_states)]

        def phi(x):
            # identity
            return x

        batch = batch_states(observations, xp=output_xp, phi=phi)
        self.assertTrue(isinstance(batch, output_xp.ndarray))
        expected_output_shape = (self.n_states,) + self.obs_shape
        self.assertEqual(batch.shape, expected_output_shape)
        testing.assert_allclose(
            batch, output_xp.ones(expected_output_shape))

    def test_batch_states_numpy_to_numpy(self):
        self._test_batch_states(np, np)

    @attr.gpu
    def test_batch_states_numpy_to_cupy(self):
        self._test_batch_states(np, chainer.cuda.cupy)

    @attr.gpu
    def test_batch_states_cupy_to_cupy(self):
        self._test_batch_states(chainer.cuda.cupy, chainer.cuda.cupy)
