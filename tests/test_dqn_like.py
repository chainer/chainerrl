from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
import os
import tempfile
import unittest

import chainer
from chainer import cuda
from chainer import optimizers
import numpy as np

import q_function
import random_seed
from envs.simple_abc import ABC
from explorers.epsilon_greedy import LinearDecayEpsilonGreedy
from explorers.epsilon_greedy import ConstantEpsilonGreedy


class _TestDQNLike(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.model_filename = os.path.join(self.tmpdir, 'model.h5')
        self.rbuf_filename = os.path.join(self.tmpdir, 'rbuf.pkl')

    def make_discrete_q_func(self, env):
        return q_function.FCSIQFunction(5, 3, 10, env.action_space.n)

    def make_continuous_q_func(self, env):
        n_dim_action = env.action_space.low.size
        return q_function.FCSIContinuousQFunction(
            5, n_dim_action, 20, 2, env.action_space)

    def make_agent(self, gpu, q_func, explorer, opt):
        raise NotImplementedError

    def _test_abc(self, gpu, discrete=True, steps=5000, load_model=False):

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

        if load_model:
            explorer = ConstantEpsilonGreedy(0.1, random_action_func)
        else:
            explorer = LinearDecayEpsilonGreedy(
                1.0, 0.1, 1000, random_action_func)

        opt = optimizers.Adam()
        opt.setup(q_func)

        agent = self.make_agent(gpu, q_func, explorer, opt)

        if load_model:
            print('Load model from', self.model_filename)
            agent.load_model(self.model_filename)
            agent.replay_buffer.load(self.rbuf_filename)

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

        # Save
        agent.save_model(self.model_filename)
        agent.replay_buffer.save(self.rbuf_filename)

    def test_abc_discrete_gpu(self):
        self._test_abc(0, discrete=True, steps=1000)
        self._test_abc(0, discrete=True, steps=500, load_model=True)

    def test_abc_continuous_gpu(self):
        self._test_abc(0, discrete=False, steps=1000)

    def test_abc_discrete_cpu(self):
        self._test_abc(-1, discrete=True, steps=1000)

    def test_abc_continuous_cpu(self):
        self._test_abc(-1, discrete=False, steps=1000)
        self._test_abc(-1, discrete=False, steps=500, load_model=True)
