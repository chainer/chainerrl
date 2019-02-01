from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import unittest

import chainer
import chainer.functions as F
from chainer import testing
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
        np.testing.assert_almost_equal(self.qout.max.array,
                                       self.q_values.max(axis=1))

    def test_greedy_actions(self):
        self.assertIsInstance(self.qout.greedy_actions, chainer.Variable)
        np.testing.assert_equal(self.qout.greedy_actions.array,
                                self.q_values.argmax(axis=1))

    def test_evaluate_actions(self):
        sample_actions = np.random.randint(self.action_size,
                                           size=self.batch_size)
        ret = self.qout.evaluate_actions(sample_actions)
        self.assertIsInstance(ret, chainer.Variable)
        for b in range(self.batch_size):
            self.assertAlmostEqual(ret.array[b],
                                   self.q_values[b, sample_actions[b]])

    def test_compute_advantage(self):
        sample_actions = np.random.randint(self.action_size,
                                           size=self.batch_size)
        greedy_actions = self.q_values.argmax(axis=1)
        ret = self.qout.compute_advantage(sample_actions)
        self.assertIsInstance(ret, chainer.Variable)
        for b in range(self.batch_size):
            if sample_actions[b] == greedy_actions[b]:
                self.assertAlmostEqual(ret.array[b], 0)
            else:
                # An advantage to the optimal policy must be always negative
                self.assertLess(ret.array[b], 0)
                q = self.q_values[b, sample_actions[b]]
                v = self.q_values[b, greedy_actions[b]]
                adv = q - v
                self.assertAlmostEqual(ret.array[b], adv)

    def test_params(self):
        self.assertEqual(len(self.qout.params), 1)
        self.assertEqual(id(self.qout.params[0]), id(self.qout.q_values))

    def test_getitem(self):
        sliced = self.qout[:10]
        np.testing.assert_equal(sliced.q_values.array, self.q_values[:10])
        self.assertEqual(sliced.n_actions, self.action_size)
        self.assertIs(sliced.q_values_formatter,
                      self.qout.q_values_formatter)


class TestDistributionalDiscreteActionValue(unittest.TestCase):

    def setUp(self):
        self.batch_size = 30
        self.action_size = 3
        self.n_atoms = 51
        self.atom_probs = np.random.dirichlet(
            alpha=np.ones(self.n_atoms),
            size=(self.batch_size, self.action_size)).astype(np.float32)
        self.z_values = np.linspace(
            -10, 10, num=self.n_atoms, dtype=np.float32)
        self.qout = action_value.DistributionalDiscreteActionValue(
            chainer.Variable(self.atom_probs), self.z_values)
        self.q_values = (self.atom_probs * self.z_values).sum(axis=2)

    def test_max(self):
        self.assertIsInstance(self.qout.max, chainer.Variable)
        np.testing.assert_almost_equal(self.qout.max.array,
                                       self.q_values.max(axis=1))

    def test_max_as_distribution(self):
        self.assertIsInstance(
            self.qout.max_as_distribution, chainer.Variable)
        for b in range(self.batch_size):
            np.testing.assert_almost_equal(
                self.qout.max_as_distribution.array[b],
                self.atom_probs[b, self.qout.greedy_actions.array[b]])

    def test_greedy_actions(self):
        self.assertIsInstance(self.qout.greedy_actions, chainer.Variable)
        np.testing.assert_equal(self.qout.greedy_actions.array,
                                self.q_values.argmax(axis=1))

    def test_evaluate_actions(self):
        sample_actions = np.random.randint(self.action_size,
                                           size=self.batch_size)
        ret = self.qout.evaluate_actions(sample_actions)
        self.assertIsInstance(ret, chainer.Variable)
        for b in range(self.batch_size):
            self.assertAlmostEqual(ret.array[b],
                                   self.q_values[b, sample_actions[b]])

    def test_evaluate_actions_as_distribution(self):
        sample_actions = np.random.randint(self.action_size,
                                           size=self.batch_size)
        ret = self.qout.evaluate_actions_as_distribution(sample_actions)
        self.assertIsInstance(ret, chainer.Variable)
        for b in range(self.batch_size):
            np.testing.assert_almost_equal(
                ret.array[b],
                self.atom_probs[b, sample_actions[b]])

    def test_compute_advantage(self):
        sample_actions = np.random.randint(self.action_size,
                                           size=self.batch_size)
        greedy_actions = self.q_values.argmax(axis=1)
        ret = self.qout.compute_advantage(sample_actions)
        self.assertIsInstance(ret, chainer.Variable)
        for b in range(self.batch_size):
            if sample_actions[b] == greedy_actions[b]:
                self.assertAlmostEqual(ret.array[b], 0)
            else:
                # An advantage to the optimal policy must be always negative
                self.assertLess(ret.array[b], 0)
                q = self.q_values[b, sample_actions[b]]
                v = self.q_values[b, greedy_actions[b]]
                adv = q - v
                self.assertAlmostEqual(ret.array[b], adv)

    def test_params(self):
        self.assertEqual(len(self.qout.params), 1)
        self.assertIs(self.qout.params[0], self.qout.q_dist)

    def test_getitem(self):
        sliced = self.qout[:10]
        np.testing.assert_equal(sliced.q_values.array, self.q_values[:10])
        np.testing.assert_equal(sliced.z_values, self.z_values)
        np.testing.assert_equal(sliced.q_dist.array, self.atom_probs[:10])
        self.assertEqual(sliced.n_actions, self.action_size)
        self.assertIs(sliced.q_values_formatter,
                      self.qout.q_values_formatter)


class TestQuantileDiscreteActionValue(unittest.TestCase):

    def setUp(self):
        self.batch_size = 30
        self.action_size = 3
        self.n_taus = 5
        self.quantiles = np.random.normal(
            size=(self.batch_size, self.n_taus, self.action_size),
        ).astype(np.float32)
        self.av = action_value.QuantileDiscreteActionValue(
            chainer.Variable(self.quantiles))
        self.q_values = self.quantiles.mean(axis=1)

    def test_q_values(self):
        self.assertIsInstance(self.av.q_values, chainer.Variable)
        np.testing.assert_almost_equal(
            self.av.q_values.array, self.q_values)

    def test_evaluate_actions_as_quantiles(self):
        sample_actions = np.random.randint(self.action_size,
                                           size=self.batch_size)
        z = self.av.evaluate_actions_as_quantiles(sample_actions)
        self.assertIsInstance(z, chainer.Variable)
        for b in range(self.batch_size):
            np.testing.assert_almost_equal(
                z.array[b],
                self.quantiles[b, :, sample_actions[b]])

    def test_params(self):
        self.assertEqual(len(self.av.params), 1)
        self.assertIs(self.av.params[0], self.av.quantiles)

    def test_getitem(self):
        sliced = self.av[:10]
        np.testing.assert_almost_equal(
            sliced.q_values.array, self.q_values[:10])
        np.testing.assert_equal(sliced.quantiles.array, self.quantiles[:10])
        self.assertEqual(sliced.n_actions, self.action_size)
        self.assertIs(sliced.q_values_formatter,
                      self.av.q_values_formatter)


class TestQuadraticActionValue(unittest.TestCase):
    def test_max_unbounded(self):
        n_batch = 7
        ndim_action = 3
        mu = np.random.randn(n_batch, ndim_action).astype(np.float32)
        mat = np.broadcast_to(
            np.eye(ndim_action, dtype=np.float32)[None],
            (n_batch, ndim_action, ndim_action))
        v = np.random.randn(n_batch).astype(np.float32)
        q_out = action_value.QuadraticActionValue(
            chainer.Variable(mu),
            chainer.Variable(mat),
            chainer.Variable(v))

        v_out = q_out.max
        self.assertIsInstance(v_out, chainer.Variable)
        v_out = v_out.array

        np.testing.assert_almost_equal(v_out, v)

    def test_max_bounded(self):
        n_batch = 20
        ndim_action = 3
        mu = np.random.randn(n_batch, ndim_action).astype(np.float32)
        mat = np.broadcast_to(
            np.eye(ndim_action, dtype=np.float32)[None],
            (n_batch, ndim_action, ndim_action))
        v = np.random.randn(n_batch).astype(np.float32)
        min_action, max_action = -1.3, 1.3
        q_out = action_value.QuadraticActionValue(
            chainer.Variable(mu),
            chainer.Variable(mat),
            chainer.Variable(v),
            min_action, max_action)

        v_out = q_out.max
        self.assertIsInstance(v_out, chainer.Variable)
        v_out = v_out.array

        # If mu[i] is an valid action, v_out[i] should be v[i]
        mu_is_allowed = np.all(
            (min_action < mu) * (mu < max_action),
            axis=1)
        np.testing.assert_almost_equal(v_out[mu_is_allowed], v[mu_is_allowed])

        # Otherwise, v_out[i] should be less than v[i]
        mu_is_not_allowed = ~np.all(
            (min_action - 1e-2 < mu) * (mu < max_action + 1e-2),
            axis=1)
        np.testing.assert_array_less(
            v_out[mu_is_not_allowed],
            v[mu_is_not_allowed])

    def test_getitem(self):
        n_batch = 7
        ndim_action = 3
        mu = np.random.randn(n_batch, ndim_action).astype(np.float32)
        mat = np.broadcast_to(
            np.eye(ndim_action, dtype=np.float32)[None],
            (n_batch, ndim_action, ndim_action))
        v = np.random.randn(n_batch).astype(np.float32)
        min_action, max_action = -1, 1
        qout = action_value.QuadraticActionValue(
            chainer.Variable(mu),
            chainer.Variable(mat),
            chainer.Variable(v),
            min_action,
            max_action,
        )
        sliced = qout[:3]
        np.testing.assert_equal(sliced.mu.array, mu[:3])
        np.testing.assert_equal(sliced.mat.array, mat[:3])
        np.testing.assert_equal(sliced.v.array, v[:3])
        np.testing.assert_equal(sliced.min_action, min_action)
        np.testing.assert_equal(sliced.max_action, max_action)


@testing.parameterize(*testing.product({
    'batch_size': [1, 3],
    'action_size': [1, 2],
    'has_maximizer': [True, False],
}))
class TestSingleActionValue(unittest.TestCase):

    def setUp(self):

        def evaluator(actions):
            # negative square norm of actions
            return -F.sum(actions ** 2, axis=1)

        self.evaluator = evaluator

        if self.has_maximizer:
            def maximizer():
                return chainer.Variable(np.zeros(
                    (self.batch_size, self.action_size), dtype=np.float32))
        else:
            maximizer = None
        self.maximizer = maximizer
        self.av = action_value.SingleActionValue(
            evaluator=evaluator, maximizer=maximizer)

    def test_max(self):
        if not self.has_maximizer:
            return
        self.assertIsInstance(self.av.max, chainer.Variable)
        np.testing.assert_almost_equal(
            self.av.max.array,
            self.evaluator(self.maximizer()).array)

    def test_greedy_actions(self):
        if not self.has_maximizer:
            return
        self.assertIsInstance(self.av.greedy_actions, chainer.Variable)
        np.testing.assert_equal(self.av.greedy_actions.array,
                                self.maximizer().array)

    def test_evaluate_actions(self):
        sample_actions = np.random.randn(
            self.batch_size, self.action_size).astype(np.float32)
        ret = self.av.evaluate_actions(sample_actions)
        self.assertIsInstance(ret, chainer.Variable)
        np.testing.assert_equal(
            ret.array, self.evaluator(sample_actions).array)

    def test_compute_advantage(self):
        if not self.has_maximizer:
            return
        sample_actions = np.random.randn(
            self.batch_size, self.action_size).astype(np.float32)
        ret = self.av.compute_advantage(sample_actions)
        self.assertIsInstance(ret, chainer.Variable)
        np.testing.assert_equal(
            ret.array,
            (self.evaluator(sample_actions).array
                - self.evaluator(self.maximizer()).array))

    def test_params(self):
        # no params
        self.assertEqual(len(self.av.params), 0)

    def test_getitem(self):
        with self.assertRaises(NotImplementedError):
            self.av[:1]
