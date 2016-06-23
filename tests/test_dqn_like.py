from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import range
from future import standard_library
standard_library.install_aliases()
import unittest

import chainer
from chainer import cuda
from chainer import optimizers
# import gym
from gym import spaces
import numpy as np

import q_function
import random_seed
from envs.simple_abc import ABC
from explorers.epsilon_greedy import LinearDecayEpsilonGreedy


class _TestDQNLike(unittest.TestCase):

    def setUp(self):
        pass

    def make_discrete_q_func(self):
        return q_function.FCSIQFunction(5, 3, 10, 2)

    def make_continuous_q_func(self):
        n_dim_action = 2
        action_space = spaces.Box(
            low=np.asarray([-0.49] * n_dim_action, dtype=np.float32),
            high=np.asarray([2.49] * n_dim_action, dtype=np.float32))
        return q_function.FCSIContinuousQFunction(
            5, n_dim_action, 20, 2, action_space)

    def make_discrete_explorer(self):
        return LinearDecayEpsilonGreedy(
            1.0, 0.1, 1000, lambda: np.random.randint(3))

    def make_continuous_explorer(self):
        return LinearDecayEpsilonGreedy(
            1.0, 0.1, 1000,
            lambda: np.random.uniform(
                low=-0.49, high=2.49, size=2).astype(np.float32))

    def make_agent(self, gpu, q_func, explorer, opt):
        raise NotImplementedError

    def _test_abc(self, gpu, discrete=True):

        random_seed.set_random_seed(0)

        if discrete:
            q_func = self.make_discrete_q_func()
            explorer = self.make_discrete_explorer()
        else:
            q_func = self.make_continuous_q_func()
            explorer = self.make_continuous_explorer()

        opt = optimizers.RMSpropGraves(
            lr=1e-4, alpha=0.95, momentum=0.95, eps=1e-2)
        opt.setup(q_func)

        agent = self.make_agent(gpu, q_func, explorer, opt)

        env = ABC()

        total_r = 0
        episode_r = 0

        obs = env.reset()
        done = False
        reward = 0.0

        # Train
        for i in range(5000):
            episode_r += reward
            total_r += reward

            action = agent.act(obs, reward, done)

            if done:
                print(('i:{} explorer:{} episode_r:{}'.format(
                    i, agent.explorer, episode_r)))
                episode_r = 0
                obs = env.reset()
                done = False
                reward = 0.0
            else:
                obs, reward, done, _ = env.step(action)

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

    # def test_abc_continuous_gpu(self):
    #     self._test_abc(0, discrete=False)
    #
    # def test_abc_discrete_cpu(self):
    #     self._test_abc(-1, discrete=True)
    #
    # def test_abc_continuous_cpu(self):
    #     self._test_abc(-1, discrete=False)
