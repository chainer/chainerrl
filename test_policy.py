import unittest

import numpy as np

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
        state = np.random.rand(
            batch_size, self.n_input_channels).astype(np.float32)
        action_indices, probs = self.policy.sample_with_probability(state)
        self.assertEquals(len(action_indices), batch_size)
        self.assertEquals(probs.data.shape, (batch_size,))
        for i in xrange(batch_size):
            self.assertGreaterEqual(action_indices[i], 0)
            self.assertLess(action_indices[i], self.n_actions)
            # Probability must be strictly larger than zero because it was
            # actually sampled
            self.assertGreater(probs.data[i], 0)
            self.assertLessEqual(probs.data[i], 1)
