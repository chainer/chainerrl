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

from chainer import functions as F
from chainer import links as L
from chainer import testing

import chainerrl
from chainerrl.agents import a3c
from chainerrl.agents import pcl
from chainerrl.envs.abc import ABC
from chainerrl.optimizers import rmsprop_async
from chainerrl import policies
from chainerrl import v_function


@testing.parameterize(*(
    testing.product({
        't_max': [1],
        'use_lstm': [False],
        'episodic': [True],  # PCL doesn't work well with continuing envs
        'disable_online_update': [True, False],
        'backprop_future_values': [True],
        'train_async': [True, False],
        'batchsize': [1, 5],
    }) +
    testing.product({
        't_max': [None],
        'use_lstm': [True, False],
        'episodic': [True],
        'disable_online_update': [True, False],
        'backprop_future_values': [True],
        'train_async': [True, False],
        'batchsize': [1, 5],
    })
))
class TestPCL(unittest.TestCase):

    def setUp(self):
        self.outdir = tempfile.mkdtemp()
        logging.basicConfig(level=logging.DEBUG)

    @testing.attr.slow
    def test_abc_discrete(self):
        self._test_abc(self.t_max, self.use_lstm, episodic=self.episodic)

    def test_abc_discrete_fast(self):
        self._test_abc(self.t_max, self.use_lstm, episodic=self.episodic,
                       steps=10, require_success=False)

    @testing.attr.slow
    def test_abc_gaussian(self):
        self._test_abc(self.t_max, self.use_lstm,
                       discrete=False, episodic=self.episodic,
                       steps=100000)

    def test_abc_gaussian_fast(self):
        self._test_abc(self.t_max, self.use_lstm,
                       discrete=False, episodic=self.episodic,
                       steps=10, require_success=False)

    def _test_abc(self, t_max, use_lstm, discrete=True, episodic=True,
                  steps=100000, require_success=True):

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
        n_hidden_layers = 2
        nonlinearity = F.relu
        if use_lstm:
            if discrete:
                model = a3c.A3CSharedModel(
                    shared=L.LSTM(obs_space.low.size, n_hidden_channels),
                    pi=policies.FCSoftmaxPolicy(
                        n_hidden_channels, action_space.n,
                        n_hidden_channels=n_hidden_channels,
                        n_hidden_layers=n_hidden_layers,
                        nonlinearity=nonlinearity,
                        last_wscale=1e-2,
                    ),
                    v=v_function.FCVFunction(
                        n_hidden_channels,
                        n_hidden_channels=n_hidden_channels,
                        n_hidden_layers=n_hidden_layers,
                        nonlinearity=nonlinearity,
                        last_wscale=1e-2,
                    ),
                )
            else:
                model = a3c.A3CSharedModel(
                    shared=L.LSTM(obs_space.low.size, n_hidden_channels),
                    pi=policies.FCGaussianPolicy(
                        n_hidden_channels, action_space.low.size,
                        n_hidden_channels=n_hidden_channels,
                        n_hidden_layers=n_hidden_layers,
                        nonlinearity=nonlinearity,
                        var_wscale=1e-2,
                        var_bias=1,
                        bound_mean=True,
                        min_action=action_space.low,
                        max_action=action_space.high,
                        min_var=1e-1,
                    ),
                    v=v_function.FCVFunction(
                        n_hidden_channels,
                        n_hidden_channels=n_hidden_channels,
                        n_hidden_layers=n_hidden_layers,
                        nonlinearity=nonlinearity,
                        last_wscale=1e-2,
                    ),
                )
        else:
            if discrete:
                model = a3c.A3CSeparateModel(
                    pi=policies.FCSoftmaxPolicy(
                        obs_space.low.size, action_space.n,
                        n_hidden_channels=n_hidden_channels,
                        n_hidden_layers=n_hidden_layers,
                        nonlinearity=nonlinearity,
                        last_wscale=1e-2,
                    ),
                    v=v_function.FCVFunction(
                        obs_space.low.size,
                        n_hidden_channels=n_hidden_channels,
                        n_hidden_layers=n_hidden_layers,
                        nonlinearity=nonlinearity,
                        last_wscale=1e-2,
                    ),
                )
            else:
                model = a3c.A3CSeparateModel(
                    pi=policies.FCGaussianPolicy(
                        obs_space.low.size, action_space.low.size,
                        n_hidden_channels=n_hidden_channels,
                        n_hidden_layers=n_hidden_layers,
                        nonlinearity=nonlinearity,
                        var_wscale=1e-2,
                        var_bias=1,
                        bound_mean=True,
                        min_action=action_space.low,
                        max_action=action_space.high,
                        min_var=1e-1,
                    ),
                    v=v_function.FCVFunction(
                        obs_space.low.size,
                        n_hidden_channels=n_hidden_channels,
                        n_hidden_layers=n_hidden_layers,
                        nonlinearity=nonlinearity,
                        last_wscale=1e-2,
                    ),
                )
        eps = 1e-8 if self.backprop_future_values else 1e-1
        opt = rmsprop_async.RMSpropAsync(lr=5e-4, eps=eps, alpha=0.99)
        opt.setup(model)
        gamma = 0.5
        tau = 1e-2
        replay_buffer = chainerrl.replay_buffer.EpisodicReplayBuffer(10 ** 5)
        agent = pcl.PCL(model, opt,
                        replay_buffer=replay_buffer,
                        t_max=t_max,
                        gamma=gamma,
                        tau=tau,
                        phi=phi,
                        n_times_replay=1,
                        batchsize=self.batchsize,
                        train_async=self.train_async,
                        backprop_future_values=self.backprop_future_values,
                        act_deterministically=True)

        if self.train_async:
            with warnings.catch_warnings(record=True) as warns:
                chainerrl.experiments.train_agent_async(
                    outdir=self.outdir, processes=nproc, make_env=make_env,
                    agent=agent, steps=steps,
                    max_episode_len=2,
                    eval_interval=200,
                    eval_n_steps=None,
                    eval_n_episodes=5,
                    successful_score=1)
                assert len(warns) == 0, warns[0]
            # The agent returned by train_agent_async is not guaranteed to be
            # successful because parameters could be modified by other
            # processes after success. Thus here the successful model is loaded
            # explicitly.
            if require_success:
                agent.load(os.path.join(self.outdir, 'successful'))
        else:
            agent.process_idx = 0
            chainerrl.experiments.train_agent_with_evaluation(
                agent=agent,
                env=make_env(0, False),
                eval_env=make_env(0, True),
                outdir=self.outdir,
                steps=steps,
                train_max_episode_len=2,
                eval_interval=200,
                eval_n_steps=None,
                eval_n_episodes=5,
                successful_score=1)

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
            if require_success:
                self.assertAlmostEqual(total_r, 1)
            agent.stop_episode()
