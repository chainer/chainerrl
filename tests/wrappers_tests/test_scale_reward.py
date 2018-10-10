from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA


import unittest

from chainer import testing
import gym

import chainerrl


@testing.parameterize(*testing.product({
    'env_id': ['CartPole-v1', 'MountainCar-v0'],
    'scale': [1.0, 0.1]
}))
class TestScaleReward(unittest.TestCase):

    def test_scale_reward(self):
        env = chainerrl.wrappers.ScaleReward(
            gym.make(self.env_id), scale=self.scale)
        self.assertIsNone(env.original_reward)
        self.assertAlmostEqual(env.scale, self.scale)

        _ = env.reset()
        _, r, _, _ = env.step(env.action_space.sample())

        if self.env_id == 'CartPole-v1':
            # Original reward must be 1
            self.assertAlmostEqual(env.original_reward, 1)
            self.assertAlmostEqual(r, self.scale)
        elif self.env_id == 'MountainCar-v0':
            # Original reward must be -1
            self.assertAlmostEqual(env.original_reward, -1)
            self.assertAlmostEqual(r, -self.scale)
        else:
            assert False
