from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import unittest

import chainer
import chainer.functions as F
from chainer import testing
from chainer.testing import attr
import numpy as np

import chainerrl


@testing.parameterize(
    *testing.product({
        'in_size': [1, 5],
        'out_size': [1, 3],
        'hidden_sizes': [(), (1,), (1, 1), (7, 8)],
        'normalize_input': [True, False],
        'normalize_output': [True, False],
        'nonlinearity': ['relu', 'elu'],
        'last_wscale': [1, 1e-3],
    })
)
class TestMLPBN(unittest.TestCase):

    def _test_call(self, gpu):
        nonlinearity = getattr(F, self.nonlinearity)
        mlp = chainerrl.links.MLPBN(
            in_size=self.in_size,
            out_size=self.out_size,
            hidden_sizes=self.hidden_sizes,
            normalize_input=self.normalize_input,
            normalize_output=self.normalize_output,
            nonlinearity=nonlinearity,
            last_wscale=self.last_wscale,
        )
        batch_size = 7
        x = np.random.rand(batch_size, self.in_size).astype(np.float32)
        if gpu >= 0:
            mlp.to_gpu(gpu)
            x = chainer.cuda.to_gpu(x)
        y = mlp(x)
        self.assertEqual(y.shape, (batch_size, self.out_size))
        self.assertEqual(chainer.cuda.get_array_module(y),
                         chainer.cuda.get_array_module(x))

    def test_call_cpu(self):
        self._test_call(gpu=-1)

    @attr.gpu
    def test_call_gpu(self):
        self._test_call(gpu=0)
