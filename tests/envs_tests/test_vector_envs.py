from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import unittest

from chainer import testing
import gym
import numpy as np

import chainerrl


@testing.parameterize(*testing.product({
    'num_envs': [1, 2, 3],
    'env_id': ['CartPole-v0', 'Pendulum-v0'],
    'random_seed_offset': [0, 100],
    'vector_env_to_test': ['SerialVectorEnv', 'MultiprocessVectorEnv'],
}))
class TestSerialVectorEnv(unittest.TestCase):

    def setUp(self):
        # Init VectorEnv to test
        if self.vector_env_to_test == 'SerialVectorEnv':
            self.vec_env = chainerrl.envs.SerialVectorEnv(
                [gym.make(self.env_id) for _ in range(self.num_envs)])
        elif self.vector_env_to_test == 'MultiprocessVectorEnv':
            self.vec_env = chainerrl.envs.MultiprocessVectorEnv(
                [(lambda: gym.make(self.env_id))
                 for _ in range(self.num_envs)])
        else:
            assert False
        # Init envs to compare against
        self.envs = [gym.make(self.env_id) for _ in range(self.num_envs)]

    def tearDown(self):
        # Delete so that all the subprocesses are joined
        del self.vec_env

    def test_num_envs(self):
        self.assertEqual(self.vec_env.num_envs, self.num_envs)

    def test_action_space(self):
        self.assertEqual(self.vec_env.action_space, self.envs[0].action_space)

    def test_observation_space(self):
        self.assertEqual(
            self.vec_env.observation_space, self.envs[0].observation_space)

    def test_seed_reset_and_step(self):
        # seed
        seeds = [self.random_seed_offset + i for i in range(self.num_envs)]
        self.vec_env.seed(seeds)
        for env, seed in zip(self.envs, seeds):
            env.seed(seed)

        # reset
        obss = self.vec_env.reset()
        real_obss = [env.reset() for env in self.envs]
        np.testing.assert_allclose(obss, real_obss)

        # step
        actions = [env.action_space.sample() for env in self.envs]
        real_obss, real_rewards, real_dones, real_infos = zip(*[
            env.step(action) for env, action in zip(self.envs, actions)])
        obss, rewards, dones, infos = self.vec_env.step(actions)
        np.testing.assert_allclose(obss, real_obss)
        self.assertEqual(rewards, real_rewards)
        self.assertEqual(dones, real_dones)
        self.assertEqual(infos, real_infos)

        # reset with full mask should have no effect
        mask = np.ones(self.num_envs)
        obss = self.vec_env.reset(mask)
        np.testing.assert_allclose(obss, real_obss)

        # reset with partial mask
        mask = np.zeros(self.num_envs)
        mask[-1] = 1
        obss = self.vec_env.reset(mask)
        real_obss = list(real_obss)
        for i in range(self.num_envs):
            if not mask[i]:
                real_obss[i] = self.envs[i].reset()
        np.testing.assert_allclose(obss, real_obss)


testing.run_module(__name__, __file__)
