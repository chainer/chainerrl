from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()

import unittest

import chainer
import numpy as np

import chainerrl


def count_actions_selected_by_boltzmann(T, q_values):

    def greedy_action_func():
        raise RuntimeError('Must not be called')

    explorer = chainerrl.explorers.Boltzmann(T=T)
    action_value = chainerrl.action_value.DiscreteActionValue(q_values)

    action_count = [0] * 3

    for t in range(10000):
        a = explorer.select_action(t, greedy_action_func, action_value)
        action_count[a] += 1

    return action_count


class TestBoltzmann(unittest.TestCase):

    def test_boltzmann(self):

        # T=1
        q_values = chainer.Variable(np.asarray([[-1, 1, 0]], dtype=np.float32))
        action_count = count_actions_selected_by_boltzmann(1, q_values)
        print('T=1', action_count)
        # Actions with larger values must be selected more often
        self.assertGreater(action_count[1], action_count[2])
        self.assertGreater(action_count[2], action_count[0])

        # T=0.5
        action_count_t05 = count_actions_selected_by_boltzmann(0.5, q_values)
        print('T=0.5', action_count_t05)
        # Actions with larger values must be selected more often
        self.assertGreater(action_count_t05[1], action_count_t05[2])
        self.assertGreater(action_count_t05[2], action_count_t05[0])

        # T=0.5 must be more greedy than T=1
        self.assertGreater(action_count_t05[1], action_count[1])
