import os
import unittest
import tempfile
import multiprocessing as mp

import chainer
from chainer import optimizers
from chainer import links as L
from chainer import functions as F

import policy
import v_function
import a3c
import async
from envs.simple_abc import ABC
import run_a3c


class A3CFF(chainer.ChainList, a3c.A3CModel):

    def __init__(self, n_actions):
        self.pi = policy.FCSoftmaxPolicy(
            1, n_actions, n_hidden_channels=10, n_hidden_layers=2)
        self.v = v_function.FCVFunction(
            1, n_hidden_channels=10, n_hidden_layers=2)
        super().__init__(self.pi, self.v)

    def pi_and_v(self, state, keep_same_state=False):
        return self.pi(state), self.v(state)


class A3CLSTM(chainer.ChainList, a3c.A3CModel):

    def __init__(self, n_actions):
        self.lstm = L.LSTM(1, 10)
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


class TestA3C(unittest.TestCase):

    def setUp(self):
        pass

    def test_abc_ff(self):
        self._test_abc(1, False)
        self._test_abc(2, False)
        self._test_abc(5, False)

    def test_abc_lstm(self):
        self._test_abc(1, True)
        self._test_abc(2, True)
        self._test_abc(5, True)

    def _test_abc(self, t_max, use_lstm):

        nproc = 8
        n_actions = 3

        def make_env(process_idx, test):
            return ABC()

        def model_opt():
            if use_lstm:
                model = A3CLSTM(n_actions)
            else:
                model = A3CFF(n_actions)
            opt = optimizers.RMSprop(1e-3, eps=1e-2)
            opt.setup(model)
            return model, opt

        phi = lambda x: x

        model, opt = run_a3c.run_a3c(
            nproc, make_env, model_opt, phi, t_max, steps=40000)

        # Test
        env = ABC()
        total_r = env.reward

        def pi_func(state):
            return model.pi_and_v(state)[0]

        model.reset_state()

        while not env.is_terminal:
            pout = pi_func(chainer.Variable(
                env.state.reshape((1,) + env.state.shape)))
            # Use the most probale actions for stability of test results
            action = pout.most_probable_actions.data[0]
            print('state:', env.state, 'action:', action)
            print('probs', pout.probs.data)
            env.receive_action(action)
            total_r += env.reward
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
