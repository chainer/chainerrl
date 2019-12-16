import os
import shutil
import tempfile
import unittest

from chainer import testing
import gym
from gym.wrappers import TimeLimit

import chainerrl


@testing.parameterize(*testing.product({
    'n_episodes': [1, 2, 3, 4],
}))
class TestMonitor(unittest.TestCase):

    def test(self):
        steps = 15

        env = gym.make('CartPole-v1')
        # unwrap default TimeLimit and wrap with new one to simulate done=True
        # at step 5
        self.assertIsInstance(env, TimeLimit)
        env = env.env  # unwrap
        env = TimeLimit(env, max_episode_steps=5)  # wrap

        tmpdir = tempfile.mkdtemp()
        try:
            env = chainerrl.wrappers.Monitor(
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
                    env.reset()
                    episode_idx += 1
                    episode_len = 0
            # `env.close()` is called when `env` is gabage-collected
            # (or explicitly deleted/closed).
            del(env)
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
