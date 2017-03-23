from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()
import random

import numpy as np


class PrioritizedBuffer (object):

    def __init__(self, capacity=None, wait_priority_after_sampling=True):
        self.capacity = capacity
        self.data = []
        self.priority_tree = SumTree()
        self.data_inf = []
        self.wait_priority_after_sampling = wait_priority_after_sampling
        self.flag_wait_priority = False

    def __len__(self):
        return len(self.data) + len(self.data_inf)

    def append(self, value, priority=None):
        if self.capacity is not None and len(self) == self.capacity:
            self.pop()
        if priority is not None:
            # Append with a given priority
            i = len(self.data)
            self.data.append(value)
            self.priority_tree[i] = priority
        else:
            # Append with the highest priority
            self.data_inf.append(value)

    def _pop_random_data_inf(self):
        assert self.data_inf
        n = len(self.data_inf)
        i = random.randrange(n)
        ret = self.data_inf[i]
        self.data_inf[i] = self.data_inf[n - 1]
        self.data_inf.pop()
        return ret

    def pop(self):
        """Remove an element uniformly.

        Not prioritized.
        """
        assert len(self) > 0
        assert (not self.wait_priority_after_sampling or
                not self.flag_wait_priority)
        n = len(self.data)
        if n == 0:
            return self._pop_random_data_inf()
        i = random.randrange(0, n)
        # remove i-th
        self.priority_tree[i] = self.priority_tree[n - 1]
        del self.priority_tree[n - 1]
        ret = self.data[i]
        self.data[i] = self.data[n - 1]
        del self.data[n - 1]
        return ret

    def _prioritized_sample_indices_and_probabilities(self, n):
        assert 0 <= n <= len(self)
        indices, probabilities = self.priority_tree.prioritized_sample(
            max(0, n - len(self.data_inf)),
            remove=self.wait_priority_after_sampling)
        while len(indices) < n:
            i = len(self.data)
            e = self._pop_random_data_inf()
            self.data.append(e)
            del self.priority_tree[i]
            indices.append(i)
            probabilities.append(None)
        return indices, probabilities

    def _sample_indices_and_probabilities(self, n, uniform_ratio):
        if uniform_ratio > 0:
            # Mix uniform samples and prioritized samples
            n_uniform = np.random.binomial(n, uniform_ratio)
            n_prioritized = n - n_uniform
            pr_indices, pr_probs = \
                self._prioritized_sample_indices_and_probabilities(
                    n_prioritized)
            un_indices, un_probs = \
                self._uniform_sample_indices_and_probabilities(
                    n_uniform)
            indices = pr_indices + un_indices
            # Note: when uniform samples and prioritized samples are mixed,
            # resulting probabilities are not the true probabilities for each
            # entry to be sampled.
            probabilities = pr_probs + un_probs
            return indices, probabilities
        else:
            # Only prioritized samples
            return self._prioritized_sample_indices_and_probabilities(n)

    def sample(self, n, uniform_ratio=0):
        """Sample data along with their corresponding probabilities.

        Args:
            n (int): Number of data to sample.
            uniform_ratio (float): Ratio of uniformly sampled data.
        Returns:
            sampled data (list)
            probabitilies (list)
        """
        assert (not self.wait_priority_after_sampling or
                not self.flag_wait_priority)
        indices, probabilities = self._sample_indices_and_probabilities(
            n, uniform_ratio=uniform_ratio)
        sampled = [self.data[i] for i in indices]
        self.sampled_indices = indices
        self.flag_wait_priority = True
        return sampled, probabilities

    def set_last_priority(self, priority):
        assert (not self.wait_priority_after_sampling or
                self.flag_wait_priority)
        assert all([p > 0.0 for p in priority])
        assert len(self.sampled_indices) == len(priority)
        for i, p in zip(self.sampled_indices, priority):
            self.priority_tree[i] = p
        self.flag_wait_priority = False
        self.sampled_indices = []

    def _uniform_sample_indices_and_probabilities(self, n):
        indices = random.sample(range(len(self.data)),
                                max(0, n - len(self.data_inf)))
        probabilities = [1 / len(self)] * len(indices)
        while len(indices) < n:
            i = len(self.data)
            e = self._pop_random_data_inf()
            self.data.append(e)
            del self.priority_tree[i]
            indices.append(i)
            probabilities.append(None)
        return indices, probabilities


class SumTree (object):
    """Fast weighted sampling.

    list-like data structure
    append, update are O(log n)
    summation over an interval is O(log n) per query
    """

    def __init__(self, bd=None, l=None, r=None, s=0.0):
        # bounds, left child, right child, sum
        self.bd = bd
        self.l = l
        self.r = r
        self.s = s

    def __str__(self):
        return 'SumTree({})'.format(self._dict())

    def _dict(self):
        ret = dict()
        if self.bd is not None and self._isleaf():
            ret[self.bd[0]] = self.s
        if self.l:
            ret.update(self.l._dict())
        if self.r:
            ret.update(self.r._dict())
        return ret

    def _initdescendant(self):
        if not self._isleaf():
            c = self._center()
            self.l = SumTree(bd=(self.bd[0], c))._initdescendant()
            self.r = SumTree(bd=(c, self.bd[1]))._initdescendant()
        return self

    def _isleaf(self):
        return (self.bd[1] - self.bd[0] == 1)

    def _center(self):
        return (self.bd[0] + self.bd[1]) // 2

    def _allocindex(self, ix):
        if self.bd is None:
            self.bd = (ix, ix + 1)
        while ix >= self.bd[1]:
            r_bd = (self.bd[1], self.bd[1] * 2 - self.bd[0])
            l = SumTree(self.bd, self.l, self.r, self.s)

            r = SumTree(bd=r_bd)._initdescendant()
            self.bd = (l.bd[0], r.bd[1])
            self.l = l
            self.r = r
            # no need to update self.s because self.r.s == 0
        while ix < self.bd[0]:
            l_bd = (self.bd[0] * 2 - self.bd[1], self.bd[0])
            l = SumTree(bd=l_bd)._initdescendant()
            r = SumTree(self.bd, self.l, self.r, self.s)
            self.bd = (l.bd[0], r.bd[1])
            self.l = l
            self.r = r
            # no need to update self.s because self.l.s == 0

    def __setitem__(self, ix, val):
        self._allocindex(ix)
        self._write(ix, val)

    def _write(self, ix, val):
        if self._isleaf():
            self.s = val
        else:
            c = self._center()
            if ix < c:
                self.l._write(ix, val)
            else:
                self.r._write(ix, val)
            self.s = self.l.s + self.r.s

    def __delitem__(self, ix):
        self.__setitem__(ix, 0.0)

    def __getitem__(self, ix):
        assert self.bd[0] <= ix < self.bd[1]
        return self._read(ix)

    def _read(self, ix):
        if self._isleaf():
            return self.s
        else:
            c = self._center()
            if ix < c:
                return self.l._read(ix)
            else:
                return self.r._read(ix)

    def prioritized_sample(self, n, remove=False):
        assert n >= 0
        ixs = []
        vals = []
        total_val = self.s  # save this before it changes by removing
        for _ in range(n):
            ix, val = self._pick(random.uniform(0.0, self.s))
            ixs.append(ix)
            vals.append(val)
            self._write(ix, 0.0)
        if not remove:
            for ix, val in zip(ixs, vals):
                self._write(ix, val)
        return ixs, [v / total_val for v in vals]

    def prioritized_choice(self):
        ix, s = self._pick(random.uniform(0.0, self.s))
        return ix, s / self.s

    def _pick(self, cum):
        if self._isleaf():
            return self.bd[0], self.s
        else:
            if cum < self.l.s:
                return self.l._pick(cum)
            else:
                return self.r._pick(cum - self.l.s)
