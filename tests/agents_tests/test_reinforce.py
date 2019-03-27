from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import logging
import tempfile
import unittest

import chainer
from chainer import functions as F
from chainer import links as L
from chainer import optimizers
from chainer import testing

import chainerrl
from chainerrl.envs.abc import ABC
from chainerrl import policies


@testing.parameterize(*(
    testing.product({
        'discrete': [True, False],
        'use_lstm': [True, False],
        'batchsize': [1, 10],
        'backward_separately': [True, False],
    })
))
class TestREINFORCE(unittest.TestCase):

    def setUp(self):
        self.outdir = tempfile.mkdtemp()
        logging.basicConfig(level=logging.DEBUG)

    @testing.attr.slow
    def test_abc_cpu(self):
        self._test_abc(self.use_lstm, discrete=self.discrete)

    @testing.attr.slow
    @testing.attr.gpu
    def test_abc_gpu(self):
        self._test_abc(self.use_lstm, discrete=self.discrete, gpu=0)

    def test_abc_fast_cpu(self):
        self._test_abc(self.use_lstm, discrete=self.discrete,
                       steps=10, require_success=False)

    @testing.attr.gpu
    def test_abc_fast_gpu(self):
        self._test_abc(self.use_lstm, discrete=self.discrete,
                       steps=10, require_success=False, gpu=0)

    def _test_abc(self, use_lstm, discrete=True, steps=1000000,
                  require_success=True, gpu=-1):

        def make_env(process_idx, test):
            size = 2
            return ABC(size=size, discrete=discrete, episodic=True,
                       partially_observable=self.use_lstm,
                       deterministic=test)

        sample_env = make_env(0, False)
        action_space = sample_env.action_space
        obs_space = sample_env.observation_space

        def phi(x):
            return x

        n_hidden_channels = 20
        n_hidden_layers = 1
        nonlinearity = F.leaky_relu
        if use_lstm:
            if discrete:
                model = chainerrl.links.Sequence(
                    L.LSTM(obs_space.low.size, n_hidden_channels,
                           forget_bias_init=1),
                    policies.FCSoftmaxPolicy(
                        n_hidden_channels, action_space.n,
                        n_hidden_channels=n_hidden_channels,
                        n_hidden_layers=n_hidden_layers,
                        nonlinearity=nonlinearity),
                )
            else:
                model = chainerrl.links.Sequence(
                    L.LSTM(obs_space.low.size, n_hidden_channels,
                           forget_bias_init=1),
                    policies.FCGaussianPolicy(
                        n_hidden_channels, action_space.low.size,
                        n_hidden_channels=n_hidden_channels,
                        n_hidden_layers=n_hidden_layers,
                        bound_mean=True,
                        min_action=action_space.low,
                        max_action=action_space.high,
                        nonlinearity=nonlinearity,
                    )
                )
        else:
            if discrete:
                model = policies.FCSoftmaxPolicy(
                    obs_space.low.size, action_space.n,
                    n_hidden_channels=n_hidden_channels,
                    n_hidden_layers=n_hidden_layers,
                    nonlinearity=nonlinearity)
            else:
                model = policies.FCGaussianPolicy(
                    obs_space.low.size, action_space.low.size,
                    n_hidden_channels=n_hidden_channels,
                    n_hidden_layers=n_hidden_layers,
                    bound_mean=True,
                    min_action=action_space.low,
                    max_action=action_space.high,
                    nonlinearity=nonlinearity,
                )

        if gpu >= 0:
            chainer.cuda.get_device(gpu).use()
            model.to_gpu()

        opt = optimizers.Adam()
        opt.setup(model)
        beta = 1e-2
        agent = chainerrl.agents.REINFORCE(
            model, opt,
            beta=beta,
            phi=phi,
            batchsize=self.batchsize,
            backward_separately=self.backward_separately,
            act_deterministically=True,
        )

        chainerrl.experiments.train_agent_with_evaluation(
            agent=agent,
            env=make_env(0, False),
            eval_env=make_env(0, True),
            outdir=self.outdir,
            steps=steps,
            train_max_episode_len=2,
            eval_interval=500,
            eval_n_steps=None,
            eval_n_episodes=5,
            successful_score=1)

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
