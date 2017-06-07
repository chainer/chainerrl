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


class LinearBN(chainer.Chain):
    """Linear layer with BatchNormalization."""

    def __init__(self, in_size, out_size):
        linear = L.Linear(in_size, out_size)
        bn = L.BatchNormalization(out_size)
        bn.avg_var[:] = 1
        super().__init__(linear=linear, bn=bn)

    def __call__(self, x):
        return self.bn(self.linear(x))


class MLPBN(chainer.Chain):
    """Multi-Layer Perceptron with BatchNormalization."""

    def __init__(self, in_size, out_size, hidden_sizes, normalize_input=True,
                 normalize_output=False):
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_sizes = hidden_sizes
        self.normalize_input = normalize_input
        self.normalize_output = normalize_output

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

        if normalize_output:
            layers['output_bn'] = L.BatchNormalization(out_size)
            layers['output_bn'].avg_var[:] = 1

        super().__init__(**layers)

    def __call__(self, x):
        h = x
        assert (not chainer.config.train) or x.shape[0] > 1
        if self.normalize_input:
            h = self.input_bn(h)
        if self.hidden_sizes:
            for l in self.hidden_layers:
                h = F.relu(l(h))
        h = self.output(h)
        if self.normalize_output:
            h = self.output_bn(h)
        return h
