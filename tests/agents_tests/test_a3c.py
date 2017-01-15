from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import *
from future import standard_library
standard_library.install_aliases()
import logging
import unittest
import tempfile

import chainer
from chainer import links as L
from chainer import testing

from chainerrl import policies
from chainerrl import v_function
from chainerrl.agents import a3c
from chainerrl.envs.abc import ABC
from chainerrl.experiments.train_agent_async import train_agent_async
from chainerrl.optimizers import rmsprop_async
from chainerrl.recurrent import Recurrent


@testing.parameterize(
    *testing.product({
        't_max': [1, 2],
        'use_lstm': [False],
        'episodic': [True, False],
    }),
    *testing.product({
        't_max': [5],
        'use_lstm': [True, False],
        'episodic': [True, False],
    }),
)
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
                       steps=100000)

    def _test_abc(self, t_max, use_lstm, discrete=True, episodic=True,
                  steps=40000):

        nproc = 8

        def make_env(process_idx, test):
            return ABC(discrete=discrete, episodic=episodic or test,
                       partially_observable=self.use_lstm)

        sample_env = make_env(0, False)
        action_space = sample_env.action_space
        obs_space = sample_env.observation_space

        def phi(x):
            return x

        def make_agent(process_idx):
            n_hidden_channels = 50
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
                            n_hidden_layers=2),
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
                            n_hidden_layers=2),
                        v=v_function.FCVFunction(
                            obs_space.low.size,
                            n_hidden_channels=n_hidden_channels,
                            n_hidden_layers=2),
                    )
            opt = rmsprop_async.RMSpropAsync(lr=1e-3, eps=1e-2, alpha=0.99)
            opt.setup(model)
            gamma = 0.9
            return a3c.A3C(model, opt, t_max=t_max, gamma=gamma, beta=1e-2,
                           process_idx=process_idx, phi=phi)

        max_episode_len = None if episodic else 5

        agent = train_agent_async(
            outdir=self.outdir, processes=nproc, make_env=make_env,
            make_agent=make_agent, steps=steps,
            max_episode_len=max_episode_len,
            eval_frequency=500,
            eval_n_runs=5,
            successful_score=1)

        # The agent returned by train_agent_async is not guaranteed to be
        # successful because parameters could be modified by other processes
        # after success. Thus here the successful model is loaded explicitly.
        agent.load(os.path.join(self.outdir, 'successful'))
        model = agent.model

        # Test
        env = make_env(0, True)
        total_r = 0
        obs = env.reset()
        done = False
        reward = 0.0

        def pi_func(state):
            return model.pi_and_v(state)[0]

        if isinstance(model, Recurrent):
            model.reset_state()

        while not done:
            pout = pi_func(chainer.Variable(
                obs.reshape((1,) + obs.shape)))
            # Use the most probale actions for stability of test results
            action = pout.most_probable.data[0]
            print('state:', obs, 'action:', action, 'pout:', pout)
            obs, reward, done, _ = env.step(action)
            total_r += reward
        self.assertAlmostEqual(total_r, 1)
