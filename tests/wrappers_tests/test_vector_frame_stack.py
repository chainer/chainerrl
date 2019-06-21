from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import functools
import mock
import unittest

from chainer import testing
import gym
import gym.spaces
import numpy as np

import chainerrl
from chainerrl.wrappers.atari_wrappers import FrameStack
from chainerrl.wrappers.atari_wrappers import LazyFrames
from chainerrl.wrappers.vector_frame_stack import VectorEnvWrapper
from chainerrl.wrappers.vector_frame_stack import VectorFrameStack


class TestVectorEnvWrapper(unittest.TestCase):

    def test(self):

        vec_env = chainerrl.envs.SerialVectorEnv(
            [mock.Mock() for _ in range(3)])

        wrapped_vec_env = VectorEnvWrapper(vec_env)

        self.assertIs(wrapped_vec_env.env, vec_env)
        self.assertIs(wrapped_vec_env.unwrapped, vec_env.unwrapped)
        self.assertIs(wrapped_vec_env.action_space, vec_env.action_space)
        self.assertIs(
            wrapped_vec_env.observation_space, vec_env.observation_space)


@testing.parameterize(*testing.product({
    'num_envs': [1, 3],
    'k': [2, 3],
}))
class TestVectorFrameStack(unittest.TestCase):

    def test(self):

        steps = 10

        # Mock env that returns atari-like frames
        def make_env(idx):
            env = mock.Mock()
            np_random = np.random.RandomState(idx)
            env.reset.side_effect = [
                np_random.rand(1, 84, 84) for _ in range(steps)]
            env.step.side_effect = [
                (
                    np_random.rand(1, 84, 84),
                    np_random.rand(),
                    bool(np_random.randint(2)),
                    {},
                )
                for _ in range(steps)]
            env.action_space = gym.spaces.Discrete(2)
            env.observation_space = gym.spaces.Box(
                low=0, high=255, shape=(1, 84, 84), dtype=np.uint8)
            return env

        # Wrap by FrameStack and MultiprocessVectorEnv
        fs_env = chainerrl.envs.MultiprocessVectorEnv(
            [functools.partial(
                FrameStack, make_env(idx), k=self.k, channel_order='chw')
             for idx, env in enumerate(range(self.num_envs))])

        # Wrap by MultiprocessVectorEnv and VectorFrameStack
        vfs_env = VectorFrameStack(
            chainerrl.envs.MultiprocessVectorEnv(
                [functools.partial(make_env, idx)
                 for idx, env in enumerate(range(self.num_envs))]),
            k=self.k, stack_axis=0)

        self.assertEqual(fs_env.action_space, vfs_env.action_space)
        self.assertEqual(fs_env.observation_space, vfs_env.observation_space)

        fs_obs = fs_env.reset()
        vfs_obs = vfs_env.reset()

        # Same LazyFrames observations
        for env_idx in range(self.num_envs):
            self.assertIsInstance(fs_obs[env_idx], LazyFrames)
            self.assertIsInstance(vfs_obs[env_idx], LazyFrames)
            np.testing.assert_allclose(
                np.asarray(fs_obs[env_idx]), np.asarray(vfs_obs[env_idx]))

        batch_action = [0] * self.num_envs
        fs_new_obs, fs_r, fs_done, _ = fs_env.step(batch_action)
        vfs_new_obs, vfs_r, vfs_done, _ = vfs_env.step(batch_action)

        # Same LazyFrames observations, but those from fs_env are copies
        # while those from vfs_env are references.
        for env_idx in range(self.num_envs):
            self.assertIsInstance(fs_new_obs[env_idx], LazyFrames)
            self.assertIsInstance(vfs_new_obs[env_idx], LazyFrames)
            np.testing.assert_allclose(
                np.asarray(fs_new_obs[env_idx]),
                np.asarray(vfs_new_obs[env_idx]))
            self.assertIsNot(
                fs_new_obs[env_idx]._frames[-2],
                fs_obs[env_idx]._frames[-1])
            self.assertIs(
                vfs_new_obs[env_idx]._frames[-2],
                vfs_obs[env_idx]._frames[-1])

        np.testing.assert_allclose(fs_r, vfs_r)
        np.testing.assert_allclose(fs_done, vfs_done)

        # Check equivalence
        for _ in range(steps - 1):
            fs_env.reset(mask=np.logical_not(fs_done))
            vfs_env.reset(mask=np.logical_not(vfs_done))
            fs_obs, fs_r, fs_done, _ = fs_env.step(batch_action)
            vfs_obs, vfs_r, vfs_done, _ = vfs_env.step(batch_action)
            for env_idx in range(self.num_envs):
                self.assertIsInstance(fs_new_obs[env_idx], LazyFrames)
                self.assertIsInstance(vfs_new_obs[env_idx], LazyFrames)
                np.testing.assert_allclose(
                    np.asarray(fs_new_obs[env_idx]),
                    np.asarray(vfs_new_obs[env_idx]))
            np.testing.assert_allclose(fs_r, vfs_r)
            np.testing.assert_allclose(fs_done, vfs_done)
