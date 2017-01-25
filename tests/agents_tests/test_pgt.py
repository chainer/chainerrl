from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()
import unittest

import chainer
from chainer import cuda
from chainer import optimizers
from chainer import testing
import numpy as np

from chainerrl.q_function import FCSAQFunction
from chainerrl.q_function import FCBNLateActionSAQFunction
from chainerrl.policies import FCGaussianPolicy
from chainerrl.policies import FCBNDeterministicPolicy
from chainerrl.misc import random_seed
from chainerrl.envs.abc import ABC
from chainerrl.explorers.epsilon_greedy import LinearDecayEpsilonGreedy
from chainerrl.explorers.epsilon_greedy import ConstantEpsilonGreedy
from chainerrl import replay_buffer
from chainerrl.agents.pgt import PGT


class TestPGT(unittest.TestCase):

    def setUp(self):
        import logging
        logging.basicConfig(level=logging.DEBUG)

    def make_model(self, env):
        ndim_obs = env.observation_space.low.size
        ndim_action = env.action_space.low.size
        policy = FCGaussianPolicy(n_input_channels=ndim_obs,
                                  n_hidden_layers=2,
                                  n_hidden_channels=50,
                                  action_size=ndim_action,
                                  min_action=env.action_space.low,
                                  max_action=env.action_space.high)

        q_func = FCBNLateActionSAQFunction(n_dim_obs=ndim_obs,
                                           n_dim_action=ndim_action,
                                           n_hidden_layers=2,
                                           n_hidden_channels=50)

        return chainer.Chain(policy=policy, q_function=q_func)

    def _test_abc(self, gpu):

        random_seed.set_random_seed(0)

        env = ABC(discrete=False)

        def random_action_func():
            a = env.action_space.sample()
            return a.astype(np.float32)

        explorer = LinearDecayEpsilonGreedy(
            1.0, 0.1, 1000, random_action_func)
        # explorer = ConstantEpsilonGreedy(0, random_action_func)

        model = self.make_model(env)
        policy = model['policy']
        q_func = model['q_function']

        actor_opt = optimizers.Adam(alpha=1e-4)
        actor_opt.setup(policy)

        critic_opt = optimizers.Adam(alpha=1e-3)
        critic_opt.setup(q_func)

        rbuf = replay_buffer.ReplayBuffer(10 ** 5)

        agent = PGT(model, actor_opt, critic_opt, rbuf, gpu=gpu, gamma=0.9,
                    explorer=explorer, replay_start_size=100,
                    target_update_method='soft', target_update_frequency=1)

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
                agent.stop_episode_and_train(obs, reward, done=done)
                print(('t:{} explorer:{} episode_r:{}'.format(
                    t, agent.explorer, episode_r)))
                episode_r = 0
                obs = env.reset()
                done = False
                reward = 0.0
            else:
                action = agent.act_and_train(obs, reward)
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
            action = policy(chainer.Variable(s), test=True).sample().data[0]
            if isinstance(action, cuda.cupy.ndarray):
                action = cuda.to_cpu(action)
            obs, reward, done, _ = env.step(action)
            total_r += reward
        self.assertAlmostEqual(total_r, 1)

    @testing.attr.slow
    def test_abc_continuous_gpu(self):
        self._test_abc(0)

    @testing.attr.slow
    def test_abc_continuous_cpu(self):
        self._test_abc(-1)
