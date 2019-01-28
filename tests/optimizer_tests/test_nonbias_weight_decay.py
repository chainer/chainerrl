from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import unittest

import chainer
import chainer.links as L
from chainer import testing
import numpy as np

import chainerrl


@testing.parameterize(*testing.product(
    {
        'lr': [1.0, 0.1],
        'weight_decay_rate': [0.1, 0.05]
    }
))
class TestNonbiasWeightDecay(unittest.TestCase):

    def test(self):

        model = chainer.Chain(
            a=L.Linear(1, 2, initialW=3, initial_bias=3),
            b=chainer.Chain(c=L.Linear(2, 3, initialW=4, initial_bias=4)),
        )
        optimizer = chainer.optimizers.SGD(self.lr)
        optimizer.setup(model)
        optimizer.add_hook(
            chainerrl.optimizers.NonbiasWeightDecay(
                rate=self.weight_decay_rate))
        optimizer.update(lambda: chainer.Variable(np.asarray(0.0)))
        decay_factor = 1 - self.lr * self.weight_decay_rate
        np.testing.assert_allclose(model.a.W.array, 3 * decay_factor)
        np.testing.assert_allclose(model.a.b.array, 3)
        np.testing.assert_allclose(model.b.c.W.array, 4 * decay_factor)
        np.testing.assert_allclose(model.b.c.b.array, 4)
