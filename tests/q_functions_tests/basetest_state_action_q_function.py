from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import unittest

import chainer
import numpy as np


class _TestSAQFunction(unittest.TestCase):

    def _test_call_given_model(self, model, gpu):
        # This method only check if a given model can receive random input
        # data and return output data with the correct interface.
        batch_size = 7
        obs = np.random.rand(batch_size, self.n_dim_obs).astype(np.float32)
        action = np.random.rand(
            batch_size, self.n_dim_action).astype(np.float32)
        if gpu >= 0:
            model.to_gpu(gpu)
            obs = chainer.cuda.to_gpu(obs)
            action = chainer.cuda.to_gpu(action)
        y = model(obs, action)
        self.assertTrue(isinstance(y, chainer.Variable))
        self.assertEqual(y.shape, (batch_size, 1))
        self.assertEqual(chainer.cuda.get_array_module(y),
                         chainer.cuda.get_array_module(obs))
