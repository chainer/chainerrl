import unittest

import numpy as np
import chainer

from chainerrl import policies


class TestFCSoftmaxPolicy(unittest.TestCase):

    def setUp(self):
        self.n_input_channels = 2
        self.n_actions = 3
        self.n_hidden_layers = 2
        self.n_hidden_channels = 100
        self.policy = policies.FCSoftmaxPolicy(
            self.n_input_channels, self.n_actions, self.n_hidden_channels,
            self.n_hidden_layers)

    def test_sample_with_probability(self):
        batch_size = 2
        state = chainer.Variable(np.random.rand(
            batch_size, self.n_input_channels).astype(np.float32))
        pout = self.policy(state)
        sample = pout.sample()
        self.assertEqual(sample.data.shape[0], batch_size)
        self.assertEqual(pout.all_prob.data.shape,
                         (batch_size, self.n_actions))
        for i in range(batch_size):
            self.assertGreaterEqual(sample.data[i], 0)
            self.assertLess(sample.data[i], self.n_actions)
            # Probability must be strictly larger than zero because it was
            # actually sampled
            for a in range(self.n_actions):
                self.assertGreater(pout.all_prob.data[i, a], 0)
                self.assertLessEqual(pout.all_prob.data[i, a], 1)
