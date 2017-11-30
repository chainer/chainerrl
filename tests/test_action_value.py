from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

import unittest

import chainer
import numpy as np

from chainerrl import action_value


class TestDiscreteActionValue(unittest.TestCase):

    def setUp(self):
        self.batch_size = 30
        self.action_size = 3
        self.q_values = np.random.normal(
            size=(self.batch_size, self.action_size)).astype(np.float32)
        self.qout = action_value.DiscreteActionValue(
            chainer.Variable(self.q_values))

    def test_max(self):
        self.assertIsInstance(self.qout.max, chainer.Variable)
        np.testing.assert_almost_equal(self.qout.max.data,
                                       self.q_values.max(axis=1))

    def test_greedy_actions(self):
        self.assertIsInstance(self.qout.greedy_actions, chainer.Variable)
        np.testing.assert_equal(self.qout.greedy_actions.data,
                                self.q_values.argmax(axis=1))

    def test_evaluate_actions(self):
        sample_actions = np.random.randint(self.action_size,
                                           size=self.batch_size)
        ret = self.qout.evaluate_actions(sample_actions)
        self.assertIsInstance(ret, chainer.Variable)
        for b in range(self.batch_size):
            self.assertAlmostEqual(ret.data[b],
                                   self.q_values[b, sample_actions[b]])

    def test_compute_advantage(self):
        sample_actions = np.random.randint(self.action_size,
                                           size=self.batch_size)
        greedy_actions = self.q_values.argmax(axis=1)
        ret = self.qout.compute_advantage(sample_actions)
        self.assertIsInstance(ret, chainer.Variable)
        for b in range(self.batch_size):
            if sample_actions[b] == greedy_actions[b]:
                self.assertAlmostEqual(ret.data[b], 0)
            else:
                # An advantage to the optimal policy must be always negative
                self.assertLess(ret.data[b], 0)
                q = self.q_values[b, sample_actions[b]]
                v = self.q_values[b, greedy_actions[b]]
                adv = q - v
                self.assertAlmostEqual(ret.data[b], adv)


class TestQuadraticActionValue(unittest.TestCase):
    def test_unbounded(self):
        n_batch = 7
        ndim_action = 3
        mu = np.random.randn(n_batch, ndim_action).astype(np.float32)
        mat = np.broadcast_to(
            np.eye(ndim_action, dtype=np.float32)[None],
            (n_batch, ndim_action, ndim_action))
        v = np.random.randn(n_batch).astype(np.float32)
        q_out = action_value.QuadraticActionValue(mu, mat, v)
        np.testing.assert_almost_equal(q_out.max.data, v)

    def test_bounded(self):
        n_batch = 20
        ndim_action = 3
        mu = np.random.randn(n_batch, ndim_action).astype(np.float32)
        mat = np.broadcast_to(
            np.eye(ndim_action, dtype=np.float32)[None],
            (n_batch, ndim_action, ndim_action))
        v = np.random.randn(n_batch).astype(np.float32)
        min_action, max_action = -1.3, 1.3
        q_out = action_value.QuadraticActionValue(
            mu, mat, v, min_action, max_action)
        v_out = q_out.max.data

        mu_is_allowed = (min_action < mu) * (mu < max_action)
        mu_is_allowed = np.all(mu_is_allowed, axis=1)

        # If mu[i] is an valid action, v_out[i] should be v[i]
        np.testing.assert_almost_equal(v_out[mu_is_allowed], v[mu_is_allowed])

        # Otherwise, v_out[i] should be less than v[i]
        np.testing.assert_array_less(
            v_out[~mu_is_allowed],
            v[~mu_is_allowed] + 1e-4)
