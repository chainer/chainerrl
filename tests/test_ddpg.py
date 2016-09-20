from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import range
from builtins import super
from future import standard_library
standard_library.install_aliases()
import os
import tempfile
import unittest

import chainer
from chainer import cuda
from chainer import optimizers
import numpy as np

from q_function import FCBNLateActionSAQFunction
from q_function import FCLSTMSAQFunction
from policy import FCBNDeterministicPolicy
from policy import FCLSTMDeterministicPolicy
import random_seed
from envs.simple_abc import ABC
from explorers.epsilon_greedy import LinearDecayEpsilonGreedy
from explorers.epsilon_greedy import ConstantEpsilonGreedy
import replay_buffer
from agents.ddpg import DDPG
from agents.ddpg import DDPGModel
from test_training import _TestTraining


class _TestDDPGOnABC(_TestTraining):

    def make_agent(self, env, gpu):
        model = self.make_model(env)
        policy = model['policy']
        q_func = model['q_function']

        actor_opt = optimizers.Adam(alpha=1e-4)
        actor_opt.setup(policy)

        critic_opt = optimizers.Adam(alpha=1e-3)
        critic_opt.setup(q_func)

        explorer = self.make_explorer(env)
        rbuf = self.make_replay_buffer(env)
        return self.make_ddpg_agent(env=env, model=model,
                                    actor_opt=actor_opt, critic_opt=critic_opt,
                                    explorer=explorer, rbuf=rbuf, gpu=gpu)

    def make_ddpg_agent(self, env, model, actor_opt, critic_opt, explorer,
                        rbuf, gpu):
        raise NotImplementedError()

    def make_explorer(self, env):
        def random_action_func():
            a = env.action_space.sample()
            if isinstance(a, np.ndarray):
                return a.astype(np.float32)
            else:
                return a
        return LinearDecayEpsilonGreedy(1.0, 0.1, 1000, random_action_func)

    def make_optimizer(self, env, q_func):
        opt = optimizers.Adam()
        opt.setup(q_func)
        return opt

    def make_replay_buffer(self, env):
        return replay_buffer.ReplayBuffer(10 ** 5)


class _TestDDPGOnContinuousPOABC(_TestDDPGOnABC):

    def make_model(self, env):
        n_dim_obs = env.observation_space.low.size
        n_dim_action = env.action_space.low.size
        policy = FCLSTMDeterministicPolicy(n_input_channels=n_dim_obs,
                                           n_hidden_layers=2,
                                           n_hidden_channels=10,
                                           action_size=n_dim_action,
                                           min_action=env.action_space.low,
                                           max_action=env.action_space.high,
                                           bound_action=True)

        q_func = FCLSTMSAQFunction(n_dim_obs=n_dim_obs,
                                   n_dim_action=n_dim_action,
                                   n_hidden_layers=2,
                                   n_hidden_channels=10)

        return DDPGModel(policy=policy, q_func=q_func)

    def make_env_and_successful_return(self):
        return ABC(discrete=False, partially_observable=True), 1

    def make_replay_buffer(self, env):
        return replay_buffer.EpisodicReplayBuffer(10 ** 5)


class _TestDDPGOnContinuousABC(_TestDDPGOnABC):

    def make_model(self, env):
        n_dim_obs = env.observation_space.low.size
        n_dim_action = env.action_space.low.size
        policy = FCBNDeterministicPolicy(n_input_channels=n_dim_obs,
                                         n_hidden_layers=2,
                                         n_hidden_channels=10,
                                         action_size=n_dim_action,
                                         min_action=env.action_space.low,
                                         max_action=env.action_space.high,
                                         bound_action=True)

        q_func = FCBNLateActionSAQFunction(n_dim_obs=n_dim_obs,
                                           n_dim_action=n_dim_action,
                                           n_hidden_layers=2,
                                           n_hidden_channels=10)

        return DDPGModel(policy=policy, q_func=q_func)

    def make_env_and_successful_return(self):
        return ABC(discrete=False), 1


class TestDDPGOnContinuousPOABC(_TestDDPGOnContinuousPOABC):

    def make_ddpg_agent(self, env, model, actor_opt, critic_opt, explorer,
                        rbuf, gpu):
        return DDPG(model, actor_opt, critic_opt, rbuf, gpu=gpu, gamma=0.9,
                    explorer=explorer, replay_start_size=100,
                    target_update_method='soft', target_update_frequency=1,
                    episodic_update=True, update_frequency=1)


# class TestDDPGOnContinuousABC(_TestDDPGOnContinuousABC):
#
#     def make_ddpg_agent(self, env, model, actor_opt, critic_opt, explorer,
#                         rbuf, gpu):
#         return DDPG(model, actor_opt, critic_opt, rbuf, gpu=gpu, gamma=0.9,
#                     explorer=explorer, replay_start_size=100,
#                     target_update_method='soft', target_update_frequency=1,
#                     episodic_update=False)


# class TestDDPG(unittest.TestCase):
#
#     def setUp(self):
#         self.tmpdir = tempfile.mkdtemp()
#         self.model_filename = os.path.join(self.tmpdir, 'model.h5')
#         self.rbuf_filename = os.path.join(self.tmpdir, 'rbuf.pkl')
#
#     def make_model(self, env):
#         n_dim_action = env.action_space.low.size
#         policy = FCBNDeterministicPolicy(n_input_channels=5,
#                                          n_hidden_layers=2,
#                                          n_hidden_channels=10,
#                                          action_size=n_dim_action,
#                                          min_action=env.action_space.low,
#                                          max_action=env.action_space.high,
#                                          bound_action=True)
#
#         q_func = FCBNLateActionSAQFunction(n_dim_obs=5,
#                                            n_dim_action=n_dim_action,
#                                            n_hidden_layers=2,
#                                            n_hidden_channels=10)
#
#         return chainer.Chain(policy=policy, q_function=q_func)
#
#     def _test_abc(self, gpu, steps=5000, load_model=False):
#
#         import logging
#         logging.basicConfig(level=logging.DEBUG)
#
#         random_seed.set_random_seed(0)
#
#         env = ABC(discrete=False)
#
#         def random_action_func():
#             a = env.action_space.sample()
#             return a.astype(np.float32)
#
#         if load_model:
#             explorer = ConstantEpsilonGreedy(0.1, random_action_func)
#         else:
#             explorer = LinearDecayEpsilonGreedy(
#                 1.0, 0.1, 1000, random_action_func)
#
#         model = self.make_model(env)
#         policy = model['policy']
#         q_func = model['q_function']
#
#         actor_opt = optimizers.Adam(alpha=1e-3)
#         actor_opt.setup(policy)
#
#         critic_opt = optimizers.Adam(alpha=1e-3)
#         critic_opt.setup(q_func)
#
#         rbuf = replay_buffer.ReplayBuffer(10 ** 5)
#
#         agent = DDPG(model, actor_opt, critic_opt, rbuf, gpu=gpu, gamma=0.9,
#                      explorer=explorer, replay_start_size=100,
#                      target_update_method='soft', target_update_frequency=1)
#
#         if load_model:
#             print('Load model from', self.model_filename)
#             agent.load_model(self.model_filename)
#             print('Load replay buffer from', self.rbuf_filename)
#             rbuf.load(self.rbuf_filename)
#         print('Size of replay buffer', len(rbuf))
#
#         total_r = 0
#         episode_r = 0
#
#         obs = env.reset()
#         done = False
#         reward = 0.0
#
#         # Train
#         t = 0
#         while t < steps:
#             episode_r += reward
#             total_r += reward
#
#             if done:
#                 agent.observe_terminal(obs, reward)
#                 print(('t:{} explorer:{} episode_r:{}'.format(
#                     t, agent.explorer, episode_r)))
#                 episode_r = 0
#                 obs = env.reset()
#                 done = False
#                 reward = 0.0
#             else:
#                 action = agent.act(obs, reward)
#                 obs, reward, done, _ = env.step(action)
#                 t += 1
#
#         # Test
#         total_r = 0.0
#         obs = env.reset()
#         done = False
#         reward = 0.0
#         while not done:
#             s = np.expand_dims(obs, 0)
#             if gpu >= 0:
#                 s = cuda.to_gpu(s, device=gpu)
#             action = policy(chainer.Variable(s), test=True).data[0]
#             if isinstance(action, cuda.cupy.ndarray):
#                 action = cuda.to_cpu(action)
#             obs, reward, done, _ = env.step(action)
#             total_r += reward
#         self.assertAlmostEqual(total_r, 1)
#
#         # Save
#         agent.save_model(self.model_filename)
#         rbuf.save(self.rbuf_filename)
#
#     def test_abc_continuous_gpu(self):
#         self._test_abc(0, steps=1000, load_model=False)
#         os.remove(self.model_filename + '.target')
#         self._test_abc(0, steps=500, load_model=True)
#         self._test_abc(0, steps=0, load_model=True)
#
#     def test_abc_continuous_cpu(self):
#         self._test_abc(-1, steps=1000, load_model=False)
#         self._test_abc(-1, steps=500, load_model=True)
#         self._test_abc(-1, steps=0, load_model=True)
