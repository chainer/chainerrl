from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import super
from future import standard_library
standard_library.install_aliases()
import logging
import unittest
import tempfile

import chainer
from chainer import links as L
from chainer import functions as F
from chainer import testing

from chainerrl import policies
from chainerrl import v_function
from chainerrl.agents import a3c
from chainerrl.envs.abc import ABC
from chainerrl.experiments.train_agent_async import train_agent_async
from chainerrl.optimizers import rmsprop_async
from chainerrl.recurrent import Recurrent
from chainerrl.recurrent import RecurrentChainMixin


class A3CFF(chainer.ChainList, a3c.A3CModel):

    def __init__(self, n_actions):
        self.pi = policies.FCSoftmaxPolicy(
            5, n_actions, n_hidden_channels=10, n_hidden_layers=2)
        self.v = v_function.FCVFunction(
            5, n_hidden_channels=10, n_hidden_layers=2)
        super().__init__(self.pi, self.v)

    def pi_and_v(self, state):
        return self.pi(state), self.v(state)


class A3CLSTM(chainer.ChainList, a3c.A3CModel, RecurrentChainMixin):

    def __init__(self, n_actions):
        self.lstm = L.LSTM(5, 10)
        self.pi = policies.FCSoftmaxPolicy(
            10, n_actions, n_hidden_channels=10, n_hidden_layers=2)
        self.v = v_function.FCVFunction(
            10, n_hidden_channels=10, n_hidden_layers=2)
        super().__init__(self.lstm, self.pi, self.v)

    def pi_and_v(self, state):
        h = F.relu(self.lstm(state))
        return self.pi(h), self.v(h)


class A3CFFGaussian(chainer.ChainList, a3c.A3CModel):

    def __init__(self, n_dim_action, bound_mean, min_action, max_action):
        self.pi = policies.FCGaussianPolicy(
            5, n_dim_action, n_hidden_channels=10, n_hidden_layers=2,
            bound_mean=bound_mean, min_action=min_action,
            max_action=max_action)
        self.v = v_function.FCVFunction(
            5, n_hidden_channels=10, n_hidden_layers=2)
        super().__init__(self.pi, self.v)

    def pi_and_v(self, state):
        return self.pi(state), self.v(state)


class A3CLSTMGaussian(chainer.ChainList, a3c.A3CModel, RecurrentChainMixin):

    def __init__(self, n_dim_action, bound_mean, min_action, max_action):
        self.lstm = L.LSTM(5, 10)
        self.pi = policies.FCGaussianPolicy(
            10, n_dim_action, n_hidden_channels=10, n_hidden_layers=2,
            bound_mean=bound_mean, min_action=min_action,
            max_action=max_action)
        self.v = v_function.FCVFunction(
            10, n_hidden_channels=10, n_hidden_layers=2)
        super().__init__(self.lstm, self.pi, self.v)

    def pi_and_v(self, state):
        h = F.relu(self.lstm(state))
        return self.pi(h), self.v(h)


class TestA3C(unittest.TestCase):

    def setUp(self):
        self.outdir = tempfile.mkdtemp()
        logging.basicConfig(level=logging.DEBUG)

    @testing.attr.slow
    def test_abc_ff(self):
        self._test_abc(1, False)
        self._test_abc(2, False)
        self._test_abc(5, False)

    @testing.attr.slow
    def test_abc_lstm(self):
        self._test_abc(1, True)
        self._test_abc(2, True)
        self._test_abc(5, True)

    @testing.attr.slow
    def test_abc_ff_gaussian(self):
        self._test_abc(1, False, discrete=False)
        self._test_abc(2, False, discrete=False)
        self._test_abc(5, False, discrete=False)

    @testing.attr.slow
    def test_abc_lstm_gaussian(self):
        self._test_abc(1, True, discrete=False, steps=100000)
        self._test_abc(2, True, discrete=False, steps=100000)
        self._test_abc(5, True, discrete=False, steps=100000)

    @testing.attr.slow
    def test_abc_ff_gaussian_infinite(self):
        self._test_abc(1, False, discrete=False, episodic=False,
                       steps=100000, use_average_reward=True)
        self._test_abc(2, False, discrete=False, episodic=False,
                       steps=100000, use_average_reward=True)
        self._test_abc(5, False, discrete=False, episodic=False,
                       steps=100000, use_average_reward=True)

    @testing.attr.slow
    def test_abc_lstm_gaussian_infinite(self):
        self._test_abc(1, True, discrete=False, episodic=False,
                       steps=100000, use_average_reward=True)
        self._test_abc(2, True, discrete=False, episodic=False,
                       steps=100000, use_average_reward=True)
        self._test_abc(5, True, discrete=False, episodic=False,
                       steps=100000, use_average_reward=True)

    def _test_abc(self, t_max, use_lstm, discrete=True, episodic=True,
                  steps=40000, use_average_reward=False):

        nproc = 8

        def make_env(process_idx, test):
            return ABC(discrete=discrete, episodic=episodic or test)

        sample_env = make_env(0, False)
        action_space = sample_env.action_space

        def phi(x):
            return x

        def make_agent(process_idx):
            if use_lstm:
                if discrete:
                    model = A3CLSTM(action_space.n)
                else:
                    model = A3CLSTMGaussian(
                        action_space.low.size,
                        bound_mean=True, min_action=action_space.low,
                        max_action=action_space.high)
            else:
                if discrete:
                    model = A3CFF(action_space.n)
                else:
                    model = A3CFFGaussian(
                        action_space.low.size,
                        bound_mean=True, min_action=action_space.low,
                        max_action=action_space.high)
            opt = rmsprop_async.RMSpropAsync(lr=1e-3, eps=1e-2, alpha=0.99)
            opt.setup(model)
            gamma = 1.0 if use_average_reward else 0.9
            return a3c.A3C(model, opt, t_max=t_max, gamma=gamma, beta=1e-3,
                           process_idx=process_idx, phi=phi,
                           use_average_reward=use_average_reward)

        max_episode_len = None if episodic else 5

        agent = train_agent_async(
            outdir=self.outdir, processes=nproc, make_env=make_env,
            make_agent=make_agent, steps=steps,
            max_episode_len=max_episode_len)

        model = agent.shared_model

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
