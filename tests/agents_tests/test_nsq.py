from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import range
from future import standard_library
standard_library.install_aliases()
import copy
import logging
import tempfile
import unittest

from chainer import optimizers
from chainer import testing
import numpy as np

from chainerrl.agents import nsq
from chainerrl import q_function
from chainerrl.envs.abc import ABC
from chainerrl.experiments.train_agent_async import train_agent_async
from chainerrl.explorers.epsilon_greedy import ConstantEpsilonGreedy


class TestNSQ(unittest.TestCase):

    def setUp(self):
        self.outdir = tempfile.mkdtemp()
        logging.basicConfig(level=logging.DEBUG)

    @testing.attr.slow
    def test_abc(self):
        self._test_abc(1)
        self._test_abc(5)

    def _test_abc(self, t_max):

        nproc = 8
        n_actions = 3

        def make_env(process_idx, test):
            return ABC()

        def random_action_func():
            return np.random.randint(n_actions)

        def phi(x):
            return x

        def make_agent(process_idx):
            q_func = q_function.FCSIQFunction(5, n_actions, 10, 2)
            opt = optimizers.RMSprop(1e-3, eps=1e-2)
            opt.setup(q_func)
            explorer = ConstantEpsilonGreedy(
                process_idx / 10, random_action_func)
            return nsq.NSQ(process_idx, q_func, opt, t_max=1,
                           gamma=0.99, i_target=100, phi=phi,
                           explorer=explorer)

        agent = train_agent_async(
            outdir=self.outdir, processes=nproc, make_env=make_env,
            make_agent=make_agent, steps=10000)

        q_func = agent.shared_q_function

        # Test
        env = make_env(0, True)
        total_r = 0
        obs = env.reset()
        done = False
        r = 0.0

        while not done:
            qout = q_func(np.expand_dims(obs, 0))
            action = qout.greedy_actions.data[0]
            print(('state:', obs, 'action:', action))
            obs, r, done, _ = env.step(action)
            total_r += r
        self.assertAlmostEqual(total_r, 1)
