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

from chainerrl.agents import acer
from chainerrl.envs.abc import ABC
from chainerrl.experiments.train_agent_async import train_agent_async
from chainerrl.optimizers import rmsprop_async
from chainerrl import policies
from chainerrl import q_function
from chainerrl.replay_buffer import EpisodicReplayBuffer


@testing.parameterize(*(
    testing.product({
        't_max': [1, 2],
        'use_lstm': [False],
        'episodic': [True, False],
        'n_times_replay': [0, 8],
        'disable_online_update': [True, False],
        'use_trust_region': [True, False],
    }) +
    testing.product({
        't_max': [5],
        'use_lstm': [True, False],
        'episodic': [True, False],
        'n_times_replay': [0, 8],
        'disable_online_update': [True, False],
        'use_trust_region': [True, False],
    })
))
class TestACER(unittest.TestCase):

    def setUp(self):
        self.outdir = tempfile.mkdtemp()
        logging.basicConfig(level=logging.DEBUG)

    @testing.attr.slow
    def test_abc_discrete(self):
        self._test_abc(self.t_max, self.use_lstm, episodic=self.episodic)

    # @testing.attr.slow
    # def test_abc_gaussian(self):
    #     self._test_abc(self.t_max, self.use_lstm,
    #                    discrete=False, episodic=self.episodic,
    #                    steps=1000000)

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
        replay_buffer = EpisodicReplayBuffer(10 ** 6)
        if use_lstm:
            if discrete:
                model = acer.ACERSharedModel(
                    shared=L.LSTM(obs_space.low.size, n_hidden_channels),
                    pi=policies.FCSoftmaxPolicy(
                        n_hidden_channels, action_space.n,
                        n_hidden_channels=n_hidden_channels,
                        n_hidden_layers=2),
                    q=q_function.FCStateQFunctionWithDiscreteAction(
                        n_hidden_channels, action_space.n,
                        n_hidden_channels=n_hidden_channels,
                        n_hidden_layers=2),
                )
            else:
                model = acer.ACERSharedModel(
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
                model = acer.ACERSeparateModel(
                    pi=policies.FCSoftmaxPolicy(
                        obs_space.low.size, action_space.n,
                        n_hidden_channels=n_hidden_channels,
                        n_hidden_layers=2),
                    q=q_function.FCStateQFunctionWithDiscreteAction(
                        obs_space.low.size, action_space.n,
                        n_hidden_channels=n_hidden_channels,
                        n_hidden_layers=2),
                )
            else:
                model = acer.ACERSeparateModel(
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
        beta = 1e-2 if discrete else 1e-3
        if self.n_times_replay == 0 and self.disable_online_update:
            # At least one of them must be enabled
            self.disable_online_update = False
        agent = acer.DiscreteACER(
            model, opt, replay_buffer=replay_buffer,
            t_max=t_max, gamma=gamma, beta=beta,
            phi=phi,
            n_times_replay=self.n_times_replay,
            act_deterministically=True,
            disable_online_update=self.disable_online_update,
            replay_start_size=10,
            use_trust_region=self.use_trust_region)

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
