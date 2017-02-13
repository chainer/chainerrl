from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

import logging
import os
import tempfile
import unittest

from chainer import links as L
from chainer import testing

from chainerrl.agents import a3c
from chainerrl.envs.abc import ABC
from chainerrl.experiments.train_agent_async import train_agent_async
from chainerrl.optimizers import rmsprop_async
from chainerrl import policies
from chainerrl import v_function


@testing.parameterize(*(
    testing.product({
        't_max': [1, 2],
        'use_lstm': [False],
        'episodic': [True, False],
    }) +
    testing.product({
        't_max': [5],
        'use_lstm': [True, False],
        'episodic': [True, False],
    })
))
class TestA3C(unittest.TestCase):

    def setUp(self):
        self.outdir = tempfile.mkdtemp()
        logging.basicConfig(level=logging.DEBUG)

    @testing.attr.slow
    def test_abc_discrete(self):
        self._test_abc(self.t_max, self.use_lstm, episodic=self.episodic)

    @testing.attr.slow
    def test_abc_gaussian(self):
        self._test_abc(self.t_max, self.use_lstm,
                       discrete=False, episodic=self.episodic,
                       steps=1000000)

    def _test_abc(self, t_max, use_lstm, discrete=True, episodic=True,
                  steps=1000000):

        nproc = 8

        def make_env(process_idx, test):
            size = 2
            return ABC(size=size, discrete=discrete, episodic=episodic or test,
                       partially_observable=self.use_lstm,
                       deterministic=test)

        sample_env = make_env(0, False)
        action_space = sample_env.action_space
        obs_space = sample_env.observation_space

        def phi(x):
            return x

        n_hidden_channels = 20
        if use_lstm:
            if discrete:
                model = a3c.A3CSharedModel(
                    shared=L.LSTM(obs_space.low.size, n_hidden_channels),
                    pi=policies.FCSoftmaxPolicy(
                        n_hidden_channels, action_space.n,
                        n_hidden_channels=n_hidden_channels,
                        n_hidden_layers=2),
                    v=v_function.FCVFunction(
                        n_hidden_channels,
                        n_hidden_channels=n_hidden_channels,
                        n_hidden_layers=2),
                )
            else:
                model = a3c.A3CSharedModel(
                    shared=L.LSTM(obs_space.low.size, n_hidden_channels),
                    pi=policies.FCGaussianPolicy(
                        n_hidden_channels, action_space.low.size,
                        n_hidden_channels=n_hidden_channels,
                        n_hidden_layers=2,
                        bound_mean=True,
                        min_action=action_space.low,
                        max_action=action_space.high),
                    v=v_function.FCVFunction(
                        n_hidden_channels,
                        n_hidden_channels=n_hidden_channels,
                        n_hidden_layers=2),
                )
        else:
            if discrete:
                model = a3c.A3CSeparateModel(
                    pi=policies.FCSoftmaxPolicy(
                        obs_space.low.size, action_space.n,
                        n_hidden_channels=n_hidden_channels,
                        n_hidden_layers=2),
                    v=v_function.FCVFunction(
                        obs_space.low.size,
                        n_hidden_channels=n_hidden_channels,
                        n_hidden_layers=2),
                )
            else:
                model = a3c.A3CSeparateModel(
                    pi=policies.FCGaussianPolicy(
                        obs_space.low.size, action_space.low.size,
                        n_hidden_channels=n_hidden_channels,
                        n_hidden_layers=2,
                        bound_mean=True,
                        min_action=action_space.low,
                        max_action=action_space.high),
                    v=v_function.FCVFunction(
                        obs_space.low.size,
                        n_hidden_channels=n_hidden_channels,
                        n_hidden_layers=2),
                )
        eps = 1e-1 if discrete else 1e-2
        opt = rmsprop_async.RMSpropAsync(lr=5e-4, eps=eps, alpha=0.99)
        opt.setup(model)
        gamma = 0.9
        beta = 1e-2
        agent = a3c.A3C(model, opt, t_max=t_max, gamma=gamma, beta=beta,
                        phi=phi,
                        act_deterministically=True)

        max_episode_len = None if episodic else 2

        train_agent_async(
            outdir=self.outdir, processes=nproc, make_env=make_env,
            agent=agent, steps=steps,
            max_episode_len=max_episode_len,
            eval_frequency=500,
            eval_n_runs=5,
            successful_score=1)

        # The agent returned by train_agent_async is not guaranteed to be
        # successful because parameters could be modified by other processes
        # after success. Thus here the successful model is loaded explicitly.
        agent.load(os.path.join(self.outdir, 'successful'))
        agent.stop_episode()

        # Test
        env = make_env(0, True)
        n_test_runs = 5

        for _ in range(n_test_runs):
            total_r = 0
            obs = env.reset()
            done = False
            reward = 0.0

            while not done:
                action = agent.act(obs)
                print('state:', obs, 'action:', action)
                obs, reward, done, _ = env.step(action)
                total_r += reward
            self.assertAlmostEqual(total_r, 1)
            agent.stop_episode()
