from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

import collections
import timeit
import unittest

from chainer import testing
import numpy as np
from scipy import stats

from chainerrl.misc.collections import _sample_n_k
from chainerrl.misc.collections import RandomAccessQueue


@testing.parameterize(
    {'n': 2, 'k': 2},
    {'n': 5, 'k': 1},
    {'n': 5, 'k': 4},
    {'n': 7, 'k': 2},
    {'n': 20, 'k': 10},
    {'n': 100, 'k': 5},
)
class TestSampleNK(unittest.TestCase):
    def test_fast(self):
        self.samples = [_sample_n_k(self.n, self.k) for _ in range(200)]
        self.subtest_constraints()

    def subtest_constraints(self):
        for s in self.samples:
            self.assertEqual(len(s), self.k)

            all(0 <= x < self.n for x in s)

            # distinct
            t = np.unique(s)
            self.assertEqual(len(t), self.k)

    @testing.attr.slow
    @testing.condition.repeat_with_success_at_least(3, 2)
    def test_slow(self):
        self.samples = [_sample_n_k(self.n, self.k) for _ in range(100000)]
        self.subtest_total_counts()
        self.subtest_order_counts()

    def subtest_total_counts(self):
        if self.k == self.n:
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
            'for n in range(64, 10000): _sample_n_k(n, 64)',
            setup=setup).  repeat(repeat=10, number=1))

    @testing.attr.slow
    def _test(self):
        t = self.get_timeit(
            "from chainerrl.misc.collections import _sample_n_k")

        # faster than random.sample
        t1 = self.get_timeit("""
import random
import six
def _sample_n_k(n, k):
    return random.sample(six.moves.range(n), k)
""")
        self.assertLess(t, t1)

        # faster than np.random.choice(..., replace=False)
        t2 = self.get_timeit("""
import numpy as np
def _sample_n_k(n, k):
    return np.random.choice(n, k, replace=False)
""")
        self.assertLess(t, t2)


@testing.parameterize(*(
    testing.product({
        'maxlen': [1, 10, None],
        'init_seq': [None, [], range(5)],
    })
))
class TestRandomAccessQueue(unittest.TestCase):
    def setUp(self):
        if self.init_seq:
            self.y_queue = RandomAccessQueue(self.init_seq, maxlen=self.maxlen)
            self.t_queue = collections.deque(self.init_seq, maxlen=self.maxlen)
        else:
            self.y_queue = RandomAccessQueue(maxlen=self.maxlen)
            self.t_queue = collections.deque(maxlen=self.maxlen)

    def test1(self):
        self.check_all()

        self.check_popleft()
        self.do_append(10)
        self.check_all()

        self.check_popleft()
        self.check_popleft()
        self.do_append(11)
        self.check_all()

        # test negative indices
        n = len(self.t_queue)
        for i in range(-n, 0):
            self.check_getitem(i)

        for k in range(4):
            self.do_extend(range(k))
            self.check_all()

        for k in range(4):
            self.check_popleft()
            self.do_extend(range(k))
            self.check_all()

        for k in range(10):
            self.do_append(20 + k)
            self.check_popleft()
            self.check_popleft()
            self.check_all()

        for _ in range(100):
            self.check_popleft()

    def check_all(self):
        self.check_len()
        n = len(self.t_queue)
        for i in range(n):
            self.check_getitem(i)

    def check_len(self):
        self.assertEqual(len(self.y_queue), len(self.t_queue))

    def check_getitem(self, i):
        self.assertEqual(self.y_queue[i], self.t_queue[i])

    def do_setitem(self, i, x):
        self.y_queue[i] = x
        self.t_queue[i] = x

    def do_append(self, x):
        self.y_queue.append(x)
        self.t_queue.append(x)

    def do_extend(self, xs):
        self.y_queue.extend(xs)
        self.t_queue.extend(xs)

    def check_popleft(self):
        try:
            t = self.t_queue.popleft()
        except IndexError:
            with self.assertRaises(IndexError):
                self.y_queue.popleft()
        else:
            self.assertEqual(self.y_queue.popleft(), t)
