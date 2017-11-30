from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()

import contextlib
import os
import shutil
import tempfile
import unittest

import chainer
from chainer import testing
import numpy as np

import chainerrl


@contextlib.contextmanager
def tempdir():
    """Alternative to tempfile.TemporaryDirectory.

    tempfile.TemporaryDirectory is not available in Python 2.x.
    """
    d = tempfile.mkdtemp()
    try:
        yield d
    finally:
        shutil.rmtree(d)


_v = chainer.Variable(np.zeros(5))
_dav = chainerrl.action_value.DiscreteActionValue(
    chainer.Variable(np.zeros((5, 5))))
_qav = chainerrl.action_value.QuadraticActionValue(
    chainer.Variable(np.zeros((5, 5), dtype=np.float32)),
    chainer.Variable(np.ones((5, 5, 5), dtype=np.float32)),
    chainer.Variable(np.zeros((5, 1), dtype=np.float32)),
)
_sdis = chainerrl.distribution.SoftmaxDistribution(
    chainer.Variable(np.zeros((5, 5))))
_gdis = chainerrl.distribution.GaussianDistribution(
    chainer.Variable(np.zeros((5, 5), dtype=np.float32)),
    chainer.Variable(np.ones((5, 5), dtype=np.float32)))


@testing.parameterize(
    {'obj': [], 'expected': []},
    {'obj': (), 'expected': []},
    {'obj': _v, 'expected': [_v]},
    {'obj': _dav, 'expected': [_dav.greedy_actions, _dav.max]},
    {'obj': _qav, 'expected': [_qav.greedy_actions, _qav.max]},
    {'obj': _sdis, 'expected': list(_sdis.params)},
    {'obj': _gdis, 'expected': list(_gdis.params)},
    {'obj': [_v, _dav, _sdis], 'expected': [
        _v, _dav.greedy_actions, _dav.max] + list(_sdis.params)},
)
class TestCollectVariables(unittest.TestCase):

    def _assert_eq_var_list(self, a, b):
        # Equality between two Variable lists
        self.assertEqual(len(a), len(b))
        self.assertTrue(isinstance(a, list))
        self.assertTrue(isinstance(b, list))
        for item in a:
            self.assertTrue(isinstance(item, chainer.Variable))
        for item in b:
            self.assertTrue(isinstance(item, chainer.Variable))
        for va, vb in zip(a, b):
            self.assertEqual(id(va), id(vb))

    def test_collect_variables(self):
        vs = chainerrl.misc.collect_variables(self.obj)
        self._assert_eq_var_list(vs, self.expected)

        # Wrap by a list
        vs = chainerrl.misc.collect_variables([self.obj])
        self._assert_eq_var_list(vs, self.expected)

        # Wrap by two lists
        vs = chainerrl.misc.collect_variables([[self.obj]])
        self._assert_eq_var_list(vs, self.expected)

        # Wrap by a tuple
        vs = chainerrl.misc.collect_variables((self.obj,))
        self._assert_eq_var_list(vs, self.expected)

        # Wrap by a two tuples
        vs = chainerrl.misc.collect_variables(((self.obj,),))
        self._assert_eq_var_list(vs, self.expected)


class TestDrawComputationalGraph(unittest.TestCase):

    def test_draw_computational_graph(self):
        x = chainer.Variable(np.zeros(5))
        y = x ** 2 + chainer.Variable(np.ones(5))
        with tempdir() as d:
            filepath = os.path.join(d, 'graph')
            chainerrl.misc.draw_computational_graph(y, filepath)
            self.assertTrue(os.path.exists(filepath + '.gv'))
            if chainerrl.misc.is_graphviz_available():
                self.assertTrue(os.path.exists(filepath + '.png'))
            else:
                self.assertFalse(os.path.exists(filepath + '.png'))
