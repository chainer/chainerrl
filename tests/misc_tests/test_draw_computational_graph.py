from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()

import os
import tempfile
import unittest

import chainer
import numpy as np

import chainerrl


class TestDrawComputationalGraph(unittest.TestCase):

    def test_collect_variables(self):
        # Empty list
        vs = chainerrl.misc.collect_variables([])
        self.assertEqual(vs, [])

        vs = chainerrl.misc.collect_variables([[]])
        self.assertEqual(vs, [])

        # Empty tuple
        vs = chainerrl.misc.collect_variables(())
        self.assertEqual(vs, [])

        vs = chainerrl.misc.collect_variables(((),))
        self.assertEqual(vs, [])

        # Variable
        v = chainer.Variable(np.zeros(5))

        vs = chainerrl.misc.collect_variables(v)
        self.assertEqual(vs, [v])

        vs = chainerrl.misc.collect_variables([v])
        self.assertEqual(vs, [v])

        vs = chainerrl.misc.collect_variables([[v]])
        self.assertEqual(vs, [v])

        # ActionValue
        av = chainerrl.action_value.DiscreteActionValue(
            chainer.Variable(np.zeros((5, 5))))

        vs = chainerrl.misc.collect_variables(av)
        self.assertEqual(vs, [av.greedy_actions, av.max])

        vs = chainerrl.misc.collect_variables([av])
        self.assertEqual(vs, [av.greedy_actions, av.max])

        vs = chainerrl.misc.collect_variables([[av]])
        self.assertEqual(vs, [av.greedy_actions, av.max])

        # Distribution
        dis = chainerrl.distribution.SoftmaxDistribution(
            chainer.Variable(np.zeros((5, 5))))

        vs = chainerrl.misc.collect_variables(dis)
        self.assertEqual(vs, list(dis.params))

        vs = chainerrl.misc.collect_variables([dis])
        self.assertEqual(vs, list(dis.params))

        vs = chainerrl.misc.collect_variables([[dis]])
        self.assertEqual(vs, list(dis.params))

        # All
        vs = chainerrl.misc.collect_variables([v, av, dis])
        self.assertEqual(vs, [v, av.greedy_actions, av.max]
                         + list(dis.params))

    def test_draw_computational_graph(self):
        x = chainer.Variable(np.zeros(5))
        y = x ** 2 + chainer.Variable(np.ones(5))
        with tempfile.TemporaryDirectory() as d:
            filepath = os.path.join(d, 'graph')
            chainerrl.misc.draw_computational_graph(y, filepath)
            self.assertTrue(os.path.exists(filepath + '.gv'))
            if chainerrl.misc.is_graphviz_available():
                self.assertTrue(os.path.exists(filepath + '.png'))
            else:
                self.assertFalse(os.path.exists(filepath + '.png'))
