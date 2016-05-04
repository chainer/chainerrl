import os
import unittest
import tempfile
import multiprocessing as mp

import numpy as np
import chainer
from chainer import optimizers

import policy
import v_function
import a3c
import async
import simple_abc
import random_seed
import replay_buffer
from simple_abc import ABC


def pv_func(model, state):
    pi, v = model
    return pi(state), v(state)


class TestA3C(unittest.TestCase):

    def setUp(self):
        pass

    def test_abc(self):
        self._test_abc(1)
        self._test_abc(2)
        self._test_abc(5)

    def _test_abc(self, t_max):

        nproc = 8

        def model_opt():
            pi = policy.FCSoftmaxPolicy(
                1, n_actions, n_hidden_channels=10, n_hidden_layers=2)
            v = v_function.FCVFunction(
                1, n_hidden_channels=10, n_hidden_layers=2)
            model = chainer.ChainList(pi, v)
            opt = optimizers.RMSprop(1e-3, eps=1e-2)
            opt.setup(model)
            return model, opt

        n_actions = 3

        model, opt = model_opt()
        counter = mp.Value('i', 0)

        model_params = async.share_params_as_shared_arrays(model)
        opt_state = async.share_states_as_shared_arrays(opt)

        def run_func(process_idx):

            env = simple_abc.ABC()

            model, opt = model_opt()
            async.set_shared_params(model, model_params)
            async.set_shared_states(opt, opt_state)

            opt.setup(model)
            agent = a3c.A3C(model, pv_func, opt, t_max, 0.9, beta=1e-2)

            total_r = 0
            episode_r = 0

            for i in range(5000):

                total_r += env.reward
                episode_r += env.reward

                action = agent.act(env.state, env.reward, env.is_terminal)

                if env.is_terminal:
                    print(('i:{} counter:{} episode_r:{}'.format(
                        i, counter.value, episode_r)))
                    episode_r = 0
                    env.initialize()
                else:
                    env.receive_action(action)

                with counter.get_lock():
                    counter.value += 1

            print(('pid:{}, counter:{}, total_r:{}'.format(
                os.getpid(), counter.value, total_r)))

            return agent

        # Train
        async.run_async(nproc, run_func)

        # Test
        env = simple_abc.ABC()
        total_r = env.reward
        final_agent = a3c.A3C(model, pv_func, opt, t_max, 0.9, beta=1e-2)
        final_pi = final_agent.model[0]
        while not env.is_terminal:
            pout = final_pi(chainer.Variable(
                env.state.reshape((1,) + env.state.shape)))
            # Use the most probale actions for stability of test results
            action = pout.most_probable_actions[0]
            print(('state:', env.state, 'action:', action))
            env.receive_action(action)
            total_r += env.reward
        self.assertAlmostEqual(total_r, 1)

    def test_save_load(self):
        n_actions = 3
        pi = policy.FCSoftmaxPolicy(1, n_actions, 10, 2)
        v = v_function.FCVFunction(1, 10, 2)
        model = chainer.ChainList(pi, v)

        def pv_func(model, state):
            print(type(state))
            pi, v = model
            return pi(state), v(state)
        opt = optimizers.RMSprop(1e-3, eps=1e-2)
        opt.setup(model)
        agent = a3c.A3C(model, pv_func, opt, 1, 0.9, beta=1e-2)

        outdir = tempfile.mkdtemp()
        filename = os.path.join(outdir, 'test_a3c.h5')
        agent.save_model(filename)
        self.assertTrue(os.path.exists(filename))
        self.assertTrue(os.path.exists(filename + '.opt'))
        agent.load_model(filename)
