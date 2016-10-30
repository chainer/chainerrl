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
import numpy as np

from chainerrl import q_function
from envs.simple_abc import ABC
import run_a3c
from explorers.epsilon_greedy import ConstantEpsilonGreedy


class TestNSQ(unittest.TestCase):

    def setUp(self):
        self.outdir = tempfile.mkdtemp()
        logging.basicConfig(level=logging.DEBUG)

    def test_abc(self):
        self._test_abc(1)
        self._test_abc(5)

    def _test_abc(self, t_max):

        nproc = 8
        n_actions = 3

        def make_env(process_idx, test):
            return ABC()

        def model_opt():
            q_func = q_function.FCSIQFunction(5, n_actions, 10, 2)
            opt = optimizers.RMSprop(1e-3, eps=1e-2)
            opt.setup(q_func)
            target_q_func = copy.deepcopy(q_func)
            return (q_func, target_q_func), (opt,)

        def phi(x):
            return x

        def random_action_func():
            return np.random.randint(n_actions)

        explorers = [ConstantEpsilonGreedy(
            i / 10, random_action_func) for i in range(nproc)]

        models, opts = run_a3c.run_nsq(
            outdir=self.outdir, processes=nproc, make_env=make_env,
            model_opt=model_opt, phi=phi, t_max=t_max, steps=10000,
            explorers=explorers)

        q_func, _ = models

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
