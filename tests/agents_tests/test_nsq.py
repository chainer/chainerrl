from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import logging
import os
import tempfile
import unittest
import warnings

from chainer import testing
import numpy as np

import chainerrl
from chainerrl.agents import nsq
from chainerrl.envs.abc import ABC
from chainerrl.experiments.train_agent_async import train_agent_async
from chainerrl.misc import async_
from chainerrl.optimizers import rmsprop_async
from chainerrl.q_functions import FCLSTMStateQFunction
from chainerrl.q_functions import FCStateQFunctionWithDiscreteAction


@testing.parameterize(*(
    testing.product({
        't_max': [1],
        'use_lstm': [False],
        'episodic': [True],
        'explorer': ['boltzmann'],
    }) +
    testing.product({
        't_max': [1, 2],
        'use_lstm': [False],
        'episodic': [True, False],
        'explorer': ['epsilon_greedy'],
    }) +
    testing.product({
        't_max': [5],
        'use_lstm': [True, False],
        'episodic': [True, False],
        'explorer': ['epsilon_greedy'],
    })
))
class TestNSQ(unittest.TestCase):

    def setUp(self):
        self.outdir = tempfile.mkdtemp()
        logging.basicConfig(level=logging.DEBUG)

    @testing.attr.slow
    def test_abc(self):
        self._test_abc()

    def test_abc_fast(self):
        self._test_abc(steps=10, require_success=False)

    def _test_abc(self, steps=100000, require_success=True):

        nproc = 8

        def make_env(process_idx, test):
            return ABC(episodic=self.episodic or test,
                       partially_observable=self.use_lstm,
                       deterministic=test)

        sample_env = make_env(0, False)
        action_space = sample_env.action_space
        obs_space = sample_env.observation_space
        ndim_obs = obs_space.low.size
        n_actions = action_space.n

        def random_action_func():
            return np.random.randint(n_actions)

        def make_agent(process_idx):
            n_hidden_channels = 50
            if self.use_lstm:
                q_func = FCLSTMStateQFunction(
                    ndim_obs, n_actions,
                    n_hidden_channels=n_hidden_channels,
                    n_hidden_layers=2)
            else:
                q_func = FCStateQFunctionWithDiscreteAction(
                    ndim_obs, n_actions,
                    n_hidden_channels=n_hidden_channels,
                    n_hidden_layers=2)
            opt = rmsprop_async.RMSpropAsync(lr=1e-3, eps=1e-2, alpha=0.99)
            opt.setup(q_func)
            if self.explorer == 'epsilon_greedy':
                explorer = chainerrl.explorers.ConstantEpsilonGreedy(
                    process_idx / 10, random_action_func)
            else:
                explorer = chainerrl.explorers.Boltzmann()

            return nsq.NSQ(q_func, opt, t_max=self.t_max,
                           gamma=0.9, i_target=100,
                           explorer=explorer)

        with warnings.catch_warnings(record=True) as warns:
            agent = train_agent_async(
                outdir=self.outdir, processes=nproc, make_env=make_env,
                make_agent=make_agent, steps=steps,
                max_episode_len=5,
                eval_interval=500,
                eval_n_steps=None,
                eval_n_episodes=5,
                successful_score=1,
            )
            # There should be no AbnormalExitWarning
            self.assertEqual(
                sum(1 if issubclass(
                    w.category, async_.AbnormalExitWarning) else 0
                    for w in warns), 0)

        # The agent returned by train_agent_async is not guaranteed to be
        # successful because parameters could be modified by other processes
        # after success. Thus here the successful model is loaded explicitly.
        if require_success:
            agent.load(os.path.join(self.outdir, 'successful'))
        agent.stop_episode()

        # Test
        n_test_runs = 5
        env = make_env(0, True)
        for _ in range(n_test_runs):
            total_r = 0
            obs = env.reset()
            print('test run offset:', env._offset)
            done = False
            r = 0.0

            while not done:
                action = agent.act(obs)
                print(('state:', obs, 'action:', action))
                obs, r, done, _ = env.step(action)
                total_r += r
            if require_success:
                self.assertAlmostEqual(total_r, 1)
            agent.stop_episode()
