import collections
import random


class PrioritizedBuffer (object):
    def __init__(self, capacity=None):
        self.capacity = capacity
        self.data = []
        self.priority_tree = SumTree()
        self.data_inf = collections.deque()
        self.count_used = []
        self.flag_wait_priority = False

    def __len__(self):
        return len(self.data) + len(self.data_inf)

    def append(self, value):
        # new values are the most prioritized
        self.data_inf.append(value)

    def pop(self):
        """Remove an element uniformly.

        Not prioritized.
        """
        assert(len(self) > 0)
        assert(not self.flag_wait_priority)
        n = len(self.data)
        if n == 0:
            return self.data_inf.pop()
        i = random.randrange(0, n)
        # remove i-th
        val = self.priority_tree.read(n-1)
        self.priority_tree.write(i, val)
        self.priority_tree.write(n-1, 0.0)
        self.count_used[i] = self.count_used.pop()
        ret = self.data[i]
        self.data[i] = self.data.pop()
        return ret

    def sample(self, n):
        assert(n <= len(self.data) + len(self.data_inf))
        assert(not self.flag_wait_priority)
        indices, probabilities = self.priority_tree.prioritized_sample(
            n-len(self.data_inf), remove=True)
        sampled = []
        # There are no duplicates in sampled.
        # There may be duplicates in indices.
        #   (The last one among the duplicates is surviving.)
        for i in indices:
            sampled.append(self.data[i])
            self.count_used[i] += 1
        while len(sampled) < n and len(self.data_inf) > 0:
            i = len(self.data)
            e = self.data_inf.popleft()
            self.data.append(e)
            if self.capacity is None or i < self.capacity:
                self.priority_tree.appendindex(i)
                self.count_used.append(1)
            else:
                # overwrite randomly
                i = random.randrange(0, self.capacity)
                self.priority_tree.write(i, 0.0)
                self.count_used[i] = 1
            indices.append(i)
            probabilities.append(None)
            sampled.append(self.data[i])
        self.sampled_indices = indices
        self.flag_wait_priority = True
        return sampled, probabilities

    def set_last_priority(self, priority):
        assert(self.flag_wait_priority)
        assert(all([p > 0.0 for p in priority]))
        assert(len(self.sampled_indices) == len(priority))
        for i, p in zip(self.sampled_indices, priority):
            self.priority_tree.write(i, p)
        self.flag_wait_priority = False


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

    def appendindex(self, ix):
        if self.bd is None:
            self.bd = (0, 1)
        elif ix == self.bd[1]:
            l = SumTree(self.bd, self.l, self.r, self.s)
            r = SumTree(bd=(ix, ix*2))._initdescendant()
            self.bd = (0, ix*2)
            self.l = l
            self.r = r
            # self.s = self.l.s + self.r.s
            # ... because self.r.s == 0

    def write(self, ix, val):
        if self._isleaf():
            self.s = val
        else:
            c = self._center()
            if ix < c:
                self.l.write(ix, val)
            else:
                self.r.write(ix, val)
            self.s = self.l.s + self.r.s

    def read(self, ix):
        if self._isleaf():
            return self.s
        else:
            c = self._center()
            if ix < c:
                self.l.read(ix)
            else:
                self.r.read(ix)

    def prioritized_sample(self, n, remove=False):
        ixs = []
        vals = []
        total_val = self.s  # save this before it changes by removing
        for _ in range(n):
            ix, val = self.pick(random.uniform(0.0, self.s))
            ixs.append(ix)
            vals.append(val)
            self.write(ix, 0.0)
        if not remove:
            for ix, val in zip(ixs, vals):
                self.write(ix, val)
        return ixs, [v / total_val for v in vals]

    def prioritized_choice(self):
        ix, s = self.pick(random.uniform(0.0, self.s))
        return ix, s / self.s

    def pick(self, cum):
        if self._isleaf():
            return self.bd[0], self.s
        else:
            if cum < self.l.s:
                return self.l.pick(cum)
            else:
                return self.r.pick(cum - self.l.s)
