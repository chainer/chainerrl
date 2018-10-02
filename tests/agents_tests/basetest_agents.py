from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()  # NOQA
import os
import tempfile
import unittest

from chainer import testing

from chainerrl.envs.abc import ABC
from chainerrl.experiments.train_agent import train_agent


class _TestAgentInterface(unittest.TestCase):

    def setUp(self):
        self.env = ABC(discrete=self.discrete,
                       partially_observable=self.partially_observable,
                       episodic=self.episodic)

    def create_agent(self, env):
        raise NotImplementedError()

    def test_save_load(self):
        a = self.create_agent(self.env)
        dirname = tempfile.mkdtemp()
        a.save(dirname)
        self.assertTrue(os.path.exists(dirname))
        b = self.create_agent(self.env)
        b.load(dirname)

    def test_run_episode(self):
        agent = self.create_agent(self.env)
        done = False
        obs = self.env.reset()
        t = 0
        while t < 10 and not done:
            a = agent.act(obs)
            obs, r, done, info = self.env.step(a)
            t += 1

    @testing.attr.slow
    def test_train(self):
        agent = self.create_agent(self.env)
        train_agent(
            agent=agent,
            env=self.env,
            steps=2000,
            outdir=tempfile.mkdtemp(),
            max_episode_len=10)
