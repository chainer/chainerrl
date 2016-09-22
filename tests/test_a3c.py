from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import super
from future import standard_library
standard_library.install_aliases()
import logging
import os
import unittest
import tempfile

import chainer
from chainer import optimizers
from chainer import links as L
from chainer import functions as F

import policy
import v_function
from agents import a3c
from envs.simple_abc import ABC
import run_a3c
import rmsprop_async


class A3CFF(chainer.ChainList, a3c.A3CModel):

    def __init__(self, n_actions):
        self.pi = policy.FCSoftmaxPolicy(
            5, n_actions, n_hidden_channels=10, n_hidden_layers=2)
        self.v = v_function.FCVFunction(
            5, n_hidden_channels=10, n_hidden_layers=2)
        super().__init__(self.pi, self.v)

    def pi_and_v(self, state, keep_same_state=False):
        return self.pi(state), self.v(state)


class A3CLSTM(chainer.ChainList, a3c.A3CModel):

    def __init__(self, n_actions):
        self.lstm = L.LSTM(5, 10)
        self.pi = policy.FCSoftmaxPolicy(
            10, n_actions, n_hidden_channels=10, n_hidden_layers=2)
        self.v = v_function.FCVFunction(
            10, n_hidden_channels=10, n_hidden_layers=2)
        super().__init__(self.lstm, self.pi, self.v)

    def pi_and_v(self, state, keep_same_state=False):
        if keep_same_state:
            prev_h, prev_c = self.lstm.h, self.lstm.c
            h = F.relu(self.lstm(state))
            self.lstm.h, self.lstm.c = prev_h, prev_c
        else:
            h = F.relu(self.lstm(state))
        return self.pi(h), self.v(h)

    def reset_state(self):
        print('reset')
        self.lstm.reset_state()

    def unchain_backward(self):
        self.lstm.h.unchain_backward()
        self.lstm.c.unchain_backward()


class A3CFFGaussian(chainer.ChainList, a3c.A3CModel):

    def __init__(self, n_dim_action, bound_mean, min_action, max_action):
        self.pi = policy.FCGaussianPolicy(
            5, n_dim_action, n_hidden_channels=10, n_hidden_layers=2,
            bound_mean=bound_mean, min_action=min_action,
            max_action=max_action, clip_action=False)
        self.v = v_function.FCVFunction(
            5, n_hidden_channels=10, n_hidden_layers=2)
        super().__init__(self.pi, self.v)

    def pi_and_v(self, state, keep_same_state=False):
        return self.pi(state), self.v(state)


class A3CLSTMGaussian(chainer.ChainList, a3c.A3CModel):

    def __init__(self, n_dim_action, bound_mean, min_action, max_action):
        self.lstm = L.LSTM(5, 10)
        self.pi = policy.FCGaussianPolicy(
            10, n_dim_action, n_hidden_channels=10, n_hidden_layers=2,
            bound_mean=bound_mean, min_action=min_action,
            max_action=max_action, clip_action=False)
        self.v = v_function.FCVFunction(
            10, n_hidden_channels=10, n_hidden_layers=2)
        super().__init__(self.lstm, self.pi, self.v)

    def pi_and_v(self, state, keep_same_state=False):
        if keep_same_state:
            prev_h, prev_c = self.lstm.h, self.lstm.c
            h = F.relu(self.lstm(state))
            self.lstm.h, self.lstm.c = prev_h, prev_c
        else:
            h = F.relu(self.lstm(state))
        return self.pi(h), self.v(h)

    def reset_state(self):
        self.lstm.reset_state()

    def unchain_backward(self):
        self.lstm.h.unchain_backward()
        self.lstm.c.unchain_backward()


class TestA3C(unittest.TestCase):

    def setUp(self):
        self.outdir = tempfile.mkdtemp()
        logging.basicConfig(level=logging.DEBUG)

    def test_abc_ff(self):
        self._test_abc(1, False)
        self._test_abc(2, False)
        self._test_abc(5, False)

    def test_abc_lstm(self):
        self._test_abc(1, True)
        self._test_abc(2, True)
        self._test_abc(5, True)

    def test_abc_ff_gaussian(self):
        self._test_abc(1, False, discrete=False)
        self._test_abc(2, False, discrete=False)
        self._test_abc(5, False, discrete=False)

    def test_abc_lstm_gaussian(self):
        self._test_abc(1, True, discrete=False, steps=100000)
        self._test_abc(2, True, discrete=False, steps=100000)
        self._test_abc(5, True, discrete=False, steps=100000)

    def test_abc_ff_gaussian_infinite(self):
        self._test_abc(1, False, discrete=False, episodic=False)
        self._test_abc(2, False, discrete=False, episodic=False)
        self._test_abc(5, False, discrete=False, episodic=False)

    def test_abc_lstm_gaussian(self):
        self._test_abc(1, True, discrete=False, episodic=False, steps=400000)
        self._test_abc(2, True, discrete=False, episodic=False, steps=400000)
        self._test_abc(5, True, discrete=False, episodic=False, steps=400000)

    def _test_abc(self, t_max, use_lstm, discrete=True, episodic=True,
                  steps=40000):

        nproc = 8

        def make_env(process_idx, test):
            return ABC(discrete=discrete, episodic=episodic or test)

        sample_env = make_env(0, False)
        action_space = sample_env.action_space

        def model_opt():
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
            # opt = optimizers.RMSprop(1e-3, eps=1e-2)
            opt = rmsprop_async.RMSpropAsync(lr=1e-3, eps=1e-1, alpha=0.99)
            opt.setup(model)
            return (model,), (opt,)

        def phi(x):
            return x

        max_episode_len = None if episodic else 5

        models, opts = run_a3c.run_a3c(
            self.outdir, nproc, make_env, model_opt, phi, t_max, steps=steps,
            max_episode_len=max_episode_len)

        model, = models

        # Test
        env = make_env(0, True)
        total_r = 0
        obs = env.reset()
        done = False
        reward = 0.0

        def pi_func(state):
            return model.pi_and_v(state)[0]

        model.reset_state()

        while not done:
            pout = pi_func(chainer.Variable(
                obs.reshape((1,) + obs.shape)))
            # Use the most probale actions for stability of test results
            action = pout.most_probable_actions.data[0]
            print('state:', obs, 'action:', action)
            obs, reward, done, _ = env.step(action)
            total_r += reward
        self.assertAlmostEqual(total_r, 1)

    def _test_save_load(self, use_lstm):
        n_actions = 3
        if use_lstm:
            model = A3CFF(n_actions)
        else:
            model = A3CLSTM(n_actions)

        opt = optimizers.RMSprop(1e-3, eps=1e-2)
        opt.setup(model)
        agent = a3c.A3C(model, opt, 1, 0.9, beta=1e-2)

        outdir = tempfile.mkdtemp()
        filename = os.path.join(outdir, 'test_a3c.h5')
        agent.save_model(filename)
        self.assertTrue(os.path.exists(filename))
        self.assertTrue(os.path.exists(filename + '.opt'))
        agent.load_model(filename)

    def test_save_load_ff(self):
        self._test_save_load(False)

    def test_save_load_lstm(self):
        self._test_save_load(True)
