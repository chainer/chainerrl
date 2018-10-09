from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA
import collections

import numpy as np

from chainerrl.misc.random import sample_n_k


class PrioritizedBuffer (object):

    def __init__(self, capacity=None, wait_priority_after_sampling=True,
                 initial_max_priority=1.0):
        self.capacity = capacity
        self.data = collections.deque()
        self.priority_sums = SumTreeQueue()
        self.priority_mins = MinTreeQueue()
        self.max_priority = initial_max_priority
        self.wait_priority_after_sampling = wait_priority_after_sampling
        self.flag_wait_priority = False

    def __len__(self):
        return len(self.data)

    def append(self, value, priority=None):
        if self.capacity is not None and len(self) == self.capacity:
            self.popleft()
        if priority is None:
            # Append with the highest priority
            priority = self.max_priority

        self.data.append(value)
        self.priority_sums.append(priority)
        self.priority_mins.append(priority)

    def popleft(self):
        assert len(self) > 0
        self.priority_sums.popleft()
        self.priority_mins.popleft()
        return self.data.popleft()

    def _sample_indices_and_probabilities(self, n, uniform_ratio):
        total_priority = self.priority_sums.sum()
        min_prob = self.priority_mins.min() / total_priority
        indices = []
        priorities = []
        if uniform_ratio > 0:
            # Mix uniform samples and prioritized samples
            n_uniform = np.random.binomial(n, uniform_ratio)
            un_indices, un_priorities = \
                self.priority_sums.uniform_sample(
                    n_uniform, remove=self.wait_priority_after_sampling)
            indices.extend(un_indices)
            priorities.extend(un_priorities)
            n -= n_uniform
            min_prob = uniform_ratio / len(self) \
                + (1 - uniform_ratio) * min_prob

        pr_indices, pr_priorities = \
            self.priority_sums.prioritized_sample(
                n, remove=self.wait_priority_after_sampling)
        indices.extend(pr_indices)
        priorities.extend(pr_priorities)

        probs = [
            uniform_ratio / len(self)
            + (1 - uniform_ratio) * pri / total_priority
            for pri in priorities
        ]
        return indices, probs, min_prob

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
        indices, probabilities, min_prob = \
            self._sample_indices_and_probabilities(
                n, uniform_ratio=uniform_ratio)
        sampled = [self.data[i] for i in indices]
        self.sampled_indices = indices
        self.flag_wait_priority = True
        return sampled, probabilities, min_prob

    def set_last_priority(self, priority):
        assert (not self.wait_priority_after_sampling or
                self.flag_wait_priority)
        assert all([p > 0.0 for p in priority])
        assert len(self.sampled_indices) == len(priority)
        for i, p in zip(self.sampled_indices, priority):
            self.priority_sums[i] = p
            self.priority_mins[i] = p
            self.max_priority = max(self.max_priority, p)
        self.flag_wait_priority = False
        self.sampled_indices = []

    def _uniform_sample_indices_and_probabilities(self, n):
        indices = list(sample_n_k(
            len(self.data), n))
        probabilities = [1 / len(self)] * len(indices)
        return indices, probabilities


# Implement operations on nodes of SumTreeQueue

def _expand(node):
    if not node:
        node[:] = [], [], None


def _reduce(node, op):
    assert node
    left_node, right_node, _ = node
    parent_value = []
    if left_node:
        parent_value.append(left_node[2])
    if right_node:
        parent_value.append(right_node[2])
    if parent_value:
        node[2] = op(parent_value)
    else:
        del node[:]


def _write(index_left, index_right, node, key, value, op):
    if index_right - index_left == 1:
        if node:
            ret = node[2]
        else:
            ret = None
        if value is None:
            del node[:]
        else:
            node[:] = None, None, value
    else:
        _expand(node)
        node_left, node_right, _ = node
        index_center = (index_left + index_right) // 2
        if key < index_center:
            ret = _write(index_left, index_center, node_left, key, value, op)
        else:
            ret = _write(index_center, index_right, node_right, key, value, op)
        _reduce(node, op)
    return ret


class TreeQueue(object):
    """Queue with Binary Indexed Tree cache

    queue-like data structure
    append, update are O(log n)
    reduction over an interval is O(log n) per query
    """

    # node = left_child, right_child, value

    def __init__(self, op):
        self.length = 0
        self.op = op

    def __setitem__(self, ix, val):
        assert 0 <= ix < self.length
        assert val is not None
        self._write(ix, val)

    def _write(self, ix, val):
        ixl, ixr = self.bounds
        return _write(ixl, ixr, self.root, ix, val, self.op)

    def append(self, value):
        if self.length == 0:
            self.root = [None, None, value]
            self.bounds = 0, 1
            self.length = 1
            return

        ixl, ixr = self.bounds
        root = self.root
        if ixr == self.length:
            _, _, root_value = root
            self.root = [self.root, [], root_value]
            ixr += ixr - ixl
            self.bounds = ixl, ixr
        ret = self._write(self.length, value)
        assert ret is None
        self.length += 1

    def popleft(self):
        assert self.length > 0
        ret = self._write(0, None)
        ixl, ixr = self.bounds
        ixl -= 1
        ixr -= 1
        self.length -= 1
        if self.length == 0:
            del self.root
            del self.bounds
            return ret

        ixc = (ixl + ixr) // 2
        if ixc == 0:
            ixl = ixc
            _, self.root, _ = self.root
        self.bounds = ixl, ixr
        return ret


def _find(index_left, index_right, node, pos):
    if index_right - index_left == 1:
        return index_left
    else:
        node_left, node_right, _ = node
        index_center = (index_left + index_right) // 2
        if node_left:
            left_value = node_left[2]
        else:
            left_value = 0.0
        if pos < left_value:
            return _find(index_left, index_center, node_left, pos)
        else:
            return _find(
                index_center, index_right, node_right, pos - left_value)


class SumTreeQueue(TreeQueue):
    """Fast weighted sampling.

    queue-like data structure
    append, update are O(log n)
    summation over an interval is O(log n) per query
    """

    def __init__(self):
        super().__init__(op=sum)

    def sum(self):
        if self.length == 0:
            return 0.0
        else:
            return self.root[2]

    def uniform_sample(self, n, remove):
        assert n >= 0
        ixs = list(sample_n_k(self.length, n))
        vals = []
        if n > 0:
            for ix in ixs:
                val = self._write(ix, 0.0)
                vals.append(val)

        if not remove:
            for ix, val in zip(ixs, vals):
                self._write(ix, val)

        return ixs, vals

    def prioritized_sample(self, n, remove):
        assert n >= 0
        ixs = []
        vals = []
        if n > 0:
            root = self.root
            ixl, ixr = self.bounds
            for _ in range(n):
                ix = _find(ixl, ixr, root, np.random.uniform(0.0, root[2]))
                val = self._write(ix, 0.0)
                ixs.append(ix)
                vals.append(val)

        if not remove:
            for ix, val in zip(ixs, vals):
                self._write(ix, val)

        return ixs, vals


class MinTreeQueue(TreeQueue):

    def __init__(self):
        super().__init__(op=min)

    def min(self):
        if self.length == 0:
            return np.inf
        else:
            return self.root[2]


# Deprecated
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
            ix, val = self._pick(np.random.uniform(0.0, self.s))
            ixs.append(ix)
            vals.append(val)
            self._write(ix, 0.0)
        if not remove:
            for ix, val in zip(ixs, vals):
                self._write(ix, val)
        return ixs, [v / total_val for v in vals]

    def prioritized_choice(self):
        ix, s = self._pick(np.random.uniform(0.0, self.s))
        return ix, s / self.s

    def _pick(self, cum):
        if self._isleaf():
            return self.bd[0], self.s
        else:
            if cum < self.l.s:
                return self.l._pick(cum)
            else:
                return self.r._pick(cum - self.l.s)
