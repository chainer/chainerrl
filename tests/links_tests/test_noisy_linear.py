import unittest

import chainer
from chainer import testing
import numpy as np

from chainerrl.links import noisy_linear


@testing.parameterize(*testing.product({
    'size_args': [
        (5,),  # uninitialized from Chainer v2
        (None, 5),  # uninitialized
        (6, 5),  # initialized
    ],
    'nobias': [False, True],
}))
class TestFactorizedNoisyLinear(unittest.TestCase):
    def test_calls(self):
        l = noisy_linear.FactorizedNoisyLinear(
            *self.size_args, nobias=self.nobias)
        x_data = np.arange(12).astype(np.float32).reshape((2, 6))
        x = chainer.Variable(x_data)
        l(x)
        l(x_data + 1)
        l(x_data.reshape((2, 3, 2)))
