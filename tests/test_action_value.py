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
