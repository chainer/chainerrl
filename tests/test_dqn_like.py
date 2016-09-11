from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
import unittest

import chainer
from chainer import cuda
from chainer import optimizers
import numpy as np

import q_function
import random_seed
from envs.simple_abc import ABC
from explorers.epsilon_greedy import LinearDecayEpsilonGreedy


class _TestDQNLike(unittest.TestCase):

    def setUp(self):
        pass

    def make_discrete_q_func(self, env):
        return q_function.FCSIQFunction(5, 3, 10, env.action_space.n)

    def make_continuous_q_func(self, env):
        n_dim_action = env.action_space.low.size
        return q_function.FCSIContinuousQFunction(
            5, n_dim_action, 20, 2, env.action_space)

    def make_agent(self, gpu, q_func, explorer, opt):
        raise NotImplementedError

    def _test_abc(self, gpu, discrete=True):

        random_seed.set_random_seed(0)

        env = ABC(discrete=discrete)

        if discrete:
            q_func = self.make_discrete_q_func(env)
        else:
            q_func = self.make_continuous_q_func(env)

        def random_action_func():
            a = env.action_space.sample()
            if not discrete:
                return a.astype(np.float32)
            else:
                return a

        explorer = LinearDecayEpsilonGreedy(
            1.0, 0.1, 1000, random_action_func)

        opt = optimizers.Adam()
        opt.setup(q_func)

        agent = self.make_agent(gpu, q_func, explorer, opt)

        total_r = 0
        episode_r = 0

        obs = env.reset()
        done = False
        reward = 0.0

        # Train
        t = 0
        while t < 5000:
            episode_r += reward
            total_r += reward

            if done:
                agent.observe_terminal(obs, reward)
                print(('t:{} explorer:{} episode_r:{}'.format(
                    t, agent.explorer, episode_r)))
                episode_r = 0
                obs = env.reset()
                done = False
                reward = 0.0
            else:
                action = agent.act(obs, reward)
                obs, reward, done, _ = env.step(action)
                t += 1

        # Test
        total_r = 0.0
        obs = env.reset()
        done = False
        reward = 0.0
        while not done:
            s = np.expand_dims(obs, 0)
            if gpu >= 0:
                s = cuda.to_gpu(s, device=gpu)
            action = q_func(chainer.Variable(s)).greedy_actions.data[0]
            if isinstance(action, cuda.cupy.ndarray):
                action = cuda.to_cpu(action)
            obs, reward, done, _ = env.step(action)
            total_r += reward
        self.assertAlmostEqual(total_r, 1)

    def test_abc_discrete_gpu(self):
        self._test_abc(0, discrete=True)

    def test_abc_continuous_gpu(self):
        self._test_abc(0, discrete=False)

    def test_abc_discrete_cpu(self):
        self._test_abc(-1, discrete=True)

    def test_abc_continuous_cpu(self):
        self._test_abc(-1, discrete=False)
