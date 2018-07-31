from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()  # NOQA

import os
import tempfile
import unittest

import chainer
from chainer import testing
import numpy as np

import chainerrl


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
    {'obj': _dav, 'expected': list(_dav.params)},
    {'obj': _qav, 'expected': list(_qav.params)},
    {'obj': _sdis, 'expected': list(_sdis.params)},
    {'obj': _gdis, 'expected': list(_gdis.params)},
    {'obj': [_v, _dav, _sdis],
        'expected': [_v] + list(_dav.params) + list(_sdis.params)},
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
        dirname = tempfile.mkdtemp()
        filepath = os.path.join(dirname, 'graph')
        chainerrl.misc.draw_computational_graph(y, filepath)
        self.assertTrue(os.path.exists(filepath + '.gv'))
        if chainerrl.misc.is_graphviz_available():
            self.assertTrue(os.path.exists(filepath + '.png'))
        else:
            self.assertFalse(os.path.exists(filepath + '.png'))
