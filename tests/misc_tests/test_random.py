from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import timeit
import unittest

from chainer import testing
from chainer.testing import condition
import numpy as np
from scipy import stats

from chainerrl.misc.random import sample_n_k


@testing.parameterize(
    {'n': 2, 'k': 2},
    {'n': 5, 'k': 1},
    {'n': 5, 'k': 4},
    {'n': 7, 'k': 2},
    {'n': 20, 'k': 10},
    {'n': 100, 'k': 5},
    {'n': 1, 'k': 0},
    {'n': 0, 'k': 0},
)
class TestSampleNK(unittest.TestCase):
    def test_fast(self):
        self.samples = [sample_n_k(self.n, self.k) for _ in range(200)]
        self.subtest_constraints()

    def subtest_constraints(self):
        for s in self.samples:
            self.assertEqual(len(s), self.k)

            all(0 <= x < self.n for x in s)

            # distinct
            t = np.unique(s)
            self.assertEqual(len(t), self.k)

    @testing.attr.slow
    @condition.repeat_with_success_at_least(3, 2)
    def test_slow(self):
        self.samples = [sample_n_k(self.n, self.k) for _ in range(100000)]
        self.subtest_total_counts()
        self.subtest_order_counts()

    def subtest_total_counts(self):
        if self.k in [0, self.n]:
            return

        cnt = np.zeros(self.n)
        for s in self.samples:
            for x in s:
                cnt[x] += 1

        m = len(self.samples)

        p = self.k / self.n
        mean = m * p
        std = np.sqrt(m * p * (1 - p))

        self.subtest_normal_distrib(cnt, mean, std)

    def subtest_order_counts(self):
        if self.k < 2:
            return

        ordered_pairs = [(i, j) for j in range(self.k) for i in range(j)]
        cnt = np.zeros(len(ordered_pairs))

        for s in self.samples:
            for t, (i, j) in enumerate(ordered_pairs):
                if s[i] < s[j]:
                    cnt[t] += 1

        m = len(self.samples)

        mean = m / 2
        std = np.sqrt(m / 4)

        self.subtest_normal_distrib(cnt, mean, std)

    def subtest_normal_distrib(self, xs, mean, std):
        _, pvalue = stats.kstest(xs, 'norm', (mean, std))
        self.assertGreater(pvalue, 3e-3)


class TestSampleNKSpeed(unittest.TestCase):
    def get_timeit(self, setup):
        return min(timeit.Timer(
            'for n in range(64, 10000): sample_n_k(n, 64)',
            setup=setup).  repeat(repeat=10, number=1))

    @testing.attr.slow
    def _test(self):
        t = self.get_timeit(
            "from chainerrl.misc.random import sample_n_k")

        # faster than random.sample
        t1 = self.get_timeit("""
import random
import six
def sample_n_k(n, k):
    return random.sample(six.moves.range(n), k)
""")
        self.assertLess(t, t1)

        # faster than np.random.choice(..., replace=False)
        t2 = self.get_timeit("""
import numpy as np
def sample_n_k(n, k):
    return np.random.choice(n, k, replace=False)
""")
        self.assertLess(t, t2)
