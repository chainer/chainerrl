from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

import chainer
from chainer import functions as F
from chainer import links as L


class MLP(chainer.Chain):
    """Multi-Layer Perceptron"""

    def __init__(self, in_size, out_size, hidden_sizes):
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_sizes = hidden_sizes

        layers = {}

        if hidden_sizes:
            hidden_layers = []
            hidden_layers.append(L.Linear(in_size, hidden_sizes[0]))
            for hin, hout in zip(hidden_sizes, hidden_sizes[1:]):
                hidden_layers.append(L.Linear(hin, hout))
            layers['hidden_layers'] = chainer.ChainList(*hidden_layers)
            layers['output'] = L.Linear(hidden_sizes[-1], out_size)
        else:
            layers['output'] = L.Linear(in_size, out_size)

        super().__init__(**layers)

    def __call__(self, x, test=False):
        h = x
        if self.hidden_sizes:
            for l in self.hidden_layers:
                h = F.relu(l(h))
        return self.output(h)
