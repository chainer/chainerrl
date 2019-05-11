from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()  # NOQA
import unittest

from chainer import testing
from chainer.testing import condition
import numpy as np

from chainerrl.explorers.additive_gaussian import AdditiveGaussian


@testing.parameterize(*(
    testing.product({
        'action_size': [1, 3],
        'scale': [0, .1],
        'low': [None, -.4],
        'high': [None, .4],
    })
))
class TestAdditiveGaussian(unittest.TestCase):

    @condition.retry(3)
    def test(self):

        def greedy_action_func():
            return np.full(self.action_size, .3)

        explorer = AdditiveGaussian(self.scale, low=self.low, high=self.high)

        actions = []
        for t in range(100):
            a = explorer.select_action(t, greedy_action_func)

            if self.low is not None:
                # Clipped at lower edge
                self.assertTrue((a >= self.low).all())

            if self.high is not None:
                # Clipped at upper edge
                self.assertTrue((a <= self.high).all())

            if self.scale == 0:
                # Without noise
                self.assertTrue((a == .3).all())
            else:
                # With noise
                self.assertFalse((a == .3).all())
            actions.append(a)

        if self.low is None and self.high is None:
            np.testing.assert_allclose(
                np.mean(np.asarray(actions), axis=0), .3, atol=.1)
