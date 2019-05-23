"""Currently this script tests `chainerrl.wrappers.atari_wrappers.FrameStack`
only."""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import mock
import unittest

from chainer import testing
import gym
import gym.spaces
import numpy as np

from chainerrl.wrappers.atari_wrappers import FrameStack, LazyFrames
from chainerrl.wrappers.atari_wrappers import ScaledFloatFrame


@testing.parameterize(*testing.product({
    'dtype': [np.uint8, np.float32],
    'k': [2, 3],
}))
class TestFrameStack(unittest.TestCase):

    def test_frame_stack(self):

        steps = 10

        # Mock env that returns atari-like frames
        def make_env(idx):
            env = mock.Mock()
            np_random = np.random.RandomState(idx)
            if self.dtype is np.uint8:
                def dtyped_rand():
                    return np_random.randint(
                        low=0, high=255, size=(1, 84, 84), dtype=self.dtype)
                low, high = 0, 255
            elif self.dtype is np.float32:
                def dtyped_rand():
                    return np_random.rand(1, 84, 84).astype(self.dtype)
                low, high = -1.0, 3.14
            else:
                assert False
            env.reset.side_effect = [dtyped_rand() for _ in range(steps)]
            env.step.side_effect = [
                (
                    dtyped_rand(),
                    np_random.rand(),
                    bool(np_random.randint(2)),
                    {},
                )
                for _ in range(steps)]
            env.action_space = gym.spaces.Discrete(2)
            env.observation_space = gym.spaces.Box(
                low=low, high=high, shape=(1, 84, 84), dtype=self.dtype)
            return env

        env = make_env(42)
        fs_env = FrameStack(make_env(42), k=self.k, channel_order='chw')

        # check action/observation space
        self.assertEqual(env.action_space, fs_env.action_space)
        self.assertIs(
            env.observation_space.dtype, fs_env.observation_space.dtype)
        self.assertEqual(
            env.observation_space.low.item(0),
            fs_env.observation_space.low.item(0))
        self.assertEqual(
            env.observation_space.high.item(0),
            fs_env.observation_space.high.item(0))

        # check reset
        obs = env.reset()
        fs_obs = fs_env.reset()
        self.assertIsInstance(fs_obs, LazyFrames)
        np.testing.assert_allclose(
            obs.take(indices=0, axis=fs_env.stack_axis),
            np.asarray(fs_obs).take(indices=0, axis=fs_env.stack_axis))

        # check step
        for _ in range(steps - 1):
            action = env.action_space.sample()
            fs_action = fs_env.action_space.sample()
            obs, r, done, info = env.step(action)
            fs_obs, fs_r, fs_done, fs_info = fs_env.step(fs_action)
            self.assertIsInstance(fs_obs, LazyFrames)
            np.testing.assert_allclose(
                obs.take(indices=0, axis=fs_env.stack_axis),
                np.asarray(fs_obs).take(indices=-1, axis=fs_env.stack_axis))
            self.assertEqual(r, fs_r)
            self.assertEqual(done, fs_done)


@testing.parameterize(*testing.product({
    'dtype': [np.uint8, np.float32],
}))
class TestScaledFloatFrame(unittest.TestCase):

    def test_scaled_float_frame(self):

        steps = 10

        # Mock env that returns atari-like frames
        def make_env(idx):
            env = mock.Mock()
            np_random = np.random.RandomState(idx)
            if self.dtype is np.uint8:
                def dtyped_rand():
                    return np_random.randint(
                        low=0, high=255, size=(1, 84, 84), dtype=self.dtype)
                low, high = 0, 255
            elif self.dtype is np.float32:
                def dtyped_rand():
                    return np_random.rand(1, 84, 84).astype(self.dtype)
                low, high = -1.0, 3.14
            else:
                assert False
            env.reset.side_effect = [dtyped_rand() for _ in range(steps)]
            env.step.side_effect = [
                (
                    dtyped_rand(),
                    np_random.rand(),
                    bool(np_random.randint(2)),
                    {},
                )
                for _ in range(steps)]
            env.action_space = gym.spaces.Discrete(2)
            env.observation_space = gym.spaces.Box(
                low=low, high=high, shape=(1, 84, 84), dtype=self.dtype)
            return env

        env = make_env(42)
        s_env = ScaledFloatFrame(make_env(42))

        # check observation space
        self.assertIs(
            type(env.observation_space), type(s_env.observation_space))
        self.assertIs(s_env.observation_space.dtype, np.dtype(np.float32))
        self.assertTrue(
            s_env.observation_space.contains(s_env.observation_space.low))
        self.assertTrue(
            s_env.observation_space.contains(s_env.observation_space.high))

        # check reset
        obs = env.reset()
        s_obs = s_env.reset()
        np.testing.assert_allclose(np.array(obs) / s_env.scale, s_obs)

        # check step
        for _ in range(steps - 1):
            action = env.action_space.sample()
            s_action = s_env.action_space.sample()
            obs, r, done, info = env.step(action)
            s_obs, s_r, s_done, s_info = s_env.step(s_action)
            np.testing.assert_allclose(np.array(obs) / s_env.scale, s_obs)
            self.assertEqual(r, s_r)
            self.assertEqual(done, s_done)
