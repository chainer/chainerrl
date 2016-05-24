import unittest

import numpy as np
import chainer

import policy


class TestFCSoftmaxPolicy(unittest.TestCase):

    def setUp(self):
        self.n_input_channels = 2
        self.n_actions = 3
        self.n_hidden_layers = 2
        self.n_hidden_channels = 100
        self.policy = policy.FCSoftmaxPolicy(
            self.n_input_channels, self.n_actions, self.n_hidden_channels,
            self.n_hidden_layers)

    def test_sample_with_probability(self):
        batch_size = 2
        state = chainer.Variable(np.random.rand(
            batch_size, self.n_input_channels).astype(np.float32))
        pout = self.policy(state)
        self.assertEqual(len(pout.sampled_actions.data), batch_size)
        self.assertEqual(pout.probs.data.shape, (batch_size, self.n_actions))
        for i in range(batch_size):
            self.assertGreaterEqual(pout.sampled_actions.data[i], 0)
            self.assertLess(pout.sampled_actions.data[i], self.n_actions)
            # Probability must be strictly larger than zero because it was
            # actually sampled
            for a in range(self.n_actions):
                self.assertGreater(pout.probs.data[i, a], 0)
                self.assertLessEqual(pout.probs.data[i, a], 1)
