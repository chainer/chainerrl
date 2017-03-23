from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
import unittest

import numpy as np
import random

from chainer import testing

from chainerrl.misc import prioritized


class TestPrioritizedBuffer(unittest.TestCase):

    def test_convergence(self):
        size = 100

        buf = prioritized.PrioritizedBuffer(capacity=size)
        for x in range(size):
            buf.append(x)

        priority_init = list(range(size))
        random.shuffle(priority_init)
        count_sampled = [0] * size

        def priority(x, n):
            return priority_init[x] + 1 / count_sampled[x]

        count_none = 0
        for t in range(200):
            sampled, probabilities = buf.sample(16)
            if all([p is not None for p in probabilities]):
                priority_old = [priority(x, count_sampled[x]) for x in sampled]
                # assert: probabilities \propto priority_old
                qs = [x / y for x, y in zip(probabilities, priority_old)]
                for q in qs:
                    self.assertAlmostEqual(q, qs[0])
            else:
                count_none += 1
            for x in sampled:
                count_sampled[x] += 1
            priority_new = [priority(x, count_sampled[x]) for x in sampled]
            buf.set_last_priority(priority_new)

        for cnt in count_sampled:
            self.assertGreaterEqual(cnt, 1)
        self.assertLessEqual(count_none, size // 16 + 1)

        corr = np.corrcoef(np.array([priority_init, count_sampled]))[0, 1]
        self.assertGreater(corr, 0.8)


@testing.parameterize(
    *testing.product({
        'capacity': [1, 10],
        'wait_priority_after_sampling': [True, False],
        'initial_priority': [0.1, 1],
        'uniform_ratio': [0, 0.1, 1],
    })
)
class TestPrioritizedBufferFlooding(unittest.TestCase):

    def test_flood(self):
        buf = prioritized.PrioritizedBuffer(
            capacity=self.capacity,
            wait_priority_after_sampling=self.wait_priority_after_sampling)
        for _ in range(100):
            for x in range(self.capacity + 1):
                if self.wait_priority_after_sampling:
                    buf.append(x)
                else:
                    buf.append(x, priority=self.initial_priority)
            for _ in range(5):
                n = random.randrange(1, self.capacity + 1)
                buf.sample(n, uniform_ratio=self.uniform_ratio)
                if self.wait_priority_after_sampling:
                    buf.set_last_priority([1.0] * n)


class TestSumTree(unittest.TestCase):

    def test_read_write(self):
        t = prioritized.SumTree()
        d = dict()
        for _ in range(200):
            k = random.randint(-10, 10)
            v = random.uniform(1e-6, 1e6)
            t[k] = v
            d[k] = v

            k = random.choice(list(d.keys()))
            self.assertEqual(t[k], d[k])
