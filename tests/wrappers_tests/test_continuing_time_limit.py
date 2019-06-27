from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import mock
import unittest
import tempfile
import shutil
import os

import numpy as np
from chainer import testing
import gym
from gym.wrappers import TimeLimit

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


@testing.parameterize(*testing.product({
    'n_episodes': [1, 2, 3, 4],
}))
class TestContinuingTimeLimitMonitor(unittest.TestCase):

    def test(self):
        steps = 15

        env = gym.make('CartPole-v1')
        env = TimeLimit(env, max_episode_steps=5)  # done=True at step 5

        tmpdir = tempfile.mkdtemp()
        try:
            env = chainerrl.wrappers.ContinuingTimeLimitMonitor(
                env, directory=tmpdir, video_callable=lambda episode_id: True)
            episode_idx = 0
            episode_len = 0
            t = 0
            _ = env.reset()
            while True:
                _, _, done, info = env.step(env.action_space.sample())
                episode_len += 1
                t += 1
                if episode_idx == 1 and episode_len >= 3:
                    info['needs_reset'] = True  # simulate ContinuingTimeLimit
                if done or info.get('needs_reset', False) or t == steps:
                    if episode_idx + 1 == self.n_episodes or t == steps:
                        break
                    _ = env.reset()
                    episode_idx += 1
                    episode_len = 0
            env.close()
            # check if videos & meta files were generated
            files = os.listdir(tmpdir)
            mp4s = [f for f in files if f.endswith('.mp4')]
            metas = [f for f in files if f.endswith('.meta.json')]
            stats = [f for f in files if f.endswith('.stats.json')]
            manifests = [f for f in files if f.endswith('.manifest.json')]
            self.assertEqual(len(mp4s), self.n_episodes)
            self.assertEqual(len(metas), self.n_episodes)
            self.assertEqual(len(stats), 1)
            self.assertEqual(len(manifests), 1)

        finally:
            shutil.rmtree(tmpdir)


if __name__ == '__main__':
    unittest.main()
