import unittest
import random

import numpy as np

import q_function


class TestQFunction(unittest.TestCase):

    def setUp(self):
        pass

    def test_sample(self):
        q_func = q_function.FCSIQFunction(1, 2, 10, 2)
        N = 1000
        greedy_count = 0
        for _ in xrange(N):
            random_state = np.asarray([[random.random()]], dtype=np.float32)
            values = q_func.forward(random_state).data
            print 'q values:', values

            # Greedy
            a, q = q_func.sample_greedily_with_value(random_state)
            self.assertEquals(float(q.data), values.max())
            self.assertEquals(a[0], values.argmax())

            # Epsilon-greedy with epsilon=0, equivalent to greedy
            a, q = q_func.sample_epsilon_greedily_with_value(random_state, 0)
            self.assertEquals(float(q.data), values.max())
            self.assertEquals(a[0], values.argmax())

            # Epsilon-greedy with epsilon=0.5, which should be result in 75
            # percents of greedy actions
            a, q = q_func.sample_epsilon_greedily_with_value(random_state, 0.5)
            if a[0] == values.argmax():
                self.assertEquals(float(q.data), values.max())
                greedy_count += 1

        print 'greedy_count', greedy_count
        self.assertLess(N * 0.7, greedy_count)
        self.assertGreater(N * 0.8, greedy_count)
