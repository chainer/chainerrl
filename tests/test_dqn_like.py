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

    def make_agent(self, gpu, q_func, opt):
        raise NotImplementedError

    def _test_abc(self, gpu, q_func):

        random_seed.set_random_seed(0)

        opt = optimizers.RMSpropGraves(
            lr=2e-3, alpha=0.95, momentum=0.95, eps=1e-4)
        opt.setup(q_func)

        agent = self.make_agent(gpu, q_func, opt)

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
                print(('i:{} epsilon:{} episode_r:{}'.format(
                    i, agent.epsilon, episode_r)))
                episode_r = 0
                obs = env.reset()
                done = False
                reward = 0.0
            else:
                obs, reward, done, _ = env.step(action)

        # Test
        agent.epsilon = 0.0
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
        self._test_abc(0, self.make_discrete_q_func())

    def test_abc_continuous_gpu(self):
        self._test_abc(0, self.make_continuous_q_func())

    def test_abc_discrete_cpu(self):
        self._test_abc(-1, self.make_discrete_q_func())

    def test_abc_continuous_cpu(self):
        self._test_abc(-1, self.make_continuous_q_func())
