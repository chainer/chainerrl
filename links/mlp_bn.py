from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from builtins import super
from builtins import range
from future import standard_library
standard_library.install_aliases()
import random

import numpy as np
import chainer
from chainer import functions as F
from chainer import links as L
from chainer import cuda

from q_output import DiscreteQOutput
from q_output import ContinuousQOutput
from functions.lower_triangular_matrix import lower_triangular_matrix


class LinearBN(chainer.Chain):
    """Linear layer with BatchNormalization."""

    def __init__(self, in_size, out_size):
        linear = L.Linear(in_size, out_size)
        bn = L.BatchNormalization(out_size)
        bn.avg_var[:] = 1
        super().__init__(linear=linear, bn=bn)

    def __call__(self, x, test=False):
        return self.bn(self.linear(x), test=test)


class MLPBN(chainer.Chain):
    """Multi-Layer Perceptron with BatchNormalization."""

    def __init__(self, in_size, out_size, hidden_sizes, normalize_input=True):
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_sizes = hidden_sizes
        self.normalize_input = normalize_input

        layers = {}

        if normalize_input:
            layers['input_bn'] = L.BatchNormalization(in_size)
            layers['input_bn'].avg_var[:] = 1

        if hidden_sizes:
            hidden_layers = []
            hidden_layers.append(LinearBN(in_size, hidden_sizes[0]))
            for hin, hout in zip(hidden_sizes, hidden_sizes[1:]):
                hidden_layers.append(LinearBN(hin, hout))
            layers['hidden_layers'] = chainer.ChainList(*hidden_layers)
            layers['output'] = L.Linear(hidden_sizes[-1], out_size)
        else:
            layers['output'] = L.Linear(in_size, out_size)

        super().__init__(**layers)

    def __call__(self, x, test=False):
        h = x
        assert test or x.shape[0] > 1
        if self.normalize_input:
            h = self.input_bn(h, test=test)
        for l in self.hidden_layers:
            h = F.relu(l(h, test=test))
        return self.output(h)

