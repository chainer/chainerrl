from __future__ import print_function
from __future__ import unicode_literals
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
    'env_id': ['CartPole-v1', 'Pendulum-v0'],
    'dtype': [np.float16, np.float32, np.float64]
}))
class TestCastObservation(unittest.TestCase):

    def test_cast_observation(self):
        env = chainerrl.wrappers.CastObservation(
            gym.make(self.env_id), dtype=self.dtype)
        rtol = 1e-3 if self.dtype == np.float16 else 1e-7

        obs = env.reset()
        self.assertEqual(env.original_observation.dtype, np.float64)
        self.assertEqual(obs.dtype, self.dtype)
        np.testing.assert_allclose(env.original_observation, obs, rtol=rtol)

        obs, r, done, info = env.step(env.action_space.sample())

        self.assertEqual(env.original_observation.dtype, np.float64)
        self.assertEqual(obs.dtype, self.dtype)
        np.testing.assert_allclose(env.original_observation, obs, rtol=rtol)


@testing.parameterize(*testing.product({
    'env_id': ['CartPole-v1', 'Pendulum-v0'],
}))
class TestCastObservationToFloat32(unittest.TestCase):

    def test_cast_observation(self):
        env = chainerrl.wrappers.CastObservationToFloat32(
            gym.make(self.env_id))

        obs = env.reset()
        self.assertEqual(env.original_observation.dtype, np.float64)
        self.assertEqual(obs.dtype, np.float32)
        np.testing.assert_allclose(env.original_observation, obs)

        obs, r, done, info = env.step(env.action_space.sample())
        self.assertEqual(env.original_observation.dtype, np.float64)
        self.assertEqual(obs.dtype, np.float32)
        np.testing.assert_allclose(env.original_observation, obs)
