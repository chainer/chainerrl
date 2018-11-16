from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import mock
import unittest

from chainer import testing

import chainerrl


@testing.parameterize(*testing.product({
    'max_episode_steps': [1, 2, 3],
}))
class TestContinuingTimeLimit(unittest.TestCase):

    def test(self):
        env = mock.Mock()
        env.reset.side_effect = ['state'] * 2
        # Since info dicts are modified by the wapper, each step call needs to
        # return a new info dict.
        env.step.side_effect = [('state', 0, False, {}) for _ in range(6)]
        env = chainerrl.wrappers.ContinuingTimeLimit(
            env, max_episode_steps=self.max_episode_steps)

        env.reset()
        for t in range(2):
            _, _, done, info = env.step(0)
            if t + 1 >= self.max_episode_steps:
                self.assertTrue(info['needs_reset'])
            else:
                self.assertFalse(info.get('needs_reset', False))

        env.reset()
        for t in range(4):
            _, _, done, info = env.step(0)
            if t + 1 >= self.max_episode_steps:
                self.assertTrue(info['needs_reset'])
            else:
                self.assertFalse(info.get('needs_reset', False))
