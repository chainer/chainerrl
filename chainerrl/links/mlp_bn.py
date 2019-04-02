from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import chainer
from chainer import functions as F
from chainer.initializers import LeCunNormal
from chainer import links as L


class LinearBN(chainer.Chain):
    """Linear layer with BatchNormalization."""

    def __init__(self, in_size, out_size):
        super().__init__()
        with self.init_scope():
            self.linear = L.Linear(in_size, out_size)
            bn = L.BatchNormalization(out_size)
            bn.avg_var[:] = 1
            self.bn = bn

    def __call__(self, x):
        return self.bn(self.linear(x))


class MLPBN(chainer.Chain):
    """Multi-Layer Perceptron with Batch Normalization.

    Args:
        in_size (int): Input size.
        out_size (int): Output size.
        hidden_sizes (list of ints): Sizes of hidden channels.
        normalize_input (bool): If set to True, Batch Normalization is applied
            to inputs.
        normalize_output (bool): If set to True, Batch Normalization is applied
            to outputs.
        nonlinearity (callable): Nonlinearity between layers. It must accept a
            Variable as an argument and return a Variable with the same shape.
            Nonlinearities with learnable parameters such as PReLU are not
            supported.
        last_wscale (float): Scale of weight initialization of the last layer.
        """

    def __init__(self, in_size, out_size, hidden_sizes, normalize_input=True,
                 normalize_output=False, nonlinearity=F.relu, last_wscale=1):
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_sizes = hidden_sizes
        self.normalize_input = normalize_input
        self.normalize_output = normalize_output
        self.nonlinearity = nonlinearity

        super().__init__()
        with self.init_scope():
            if normalize_input:
                self.input_bn = L.BatchNormalization(in_size)
                self.input_bn.avg_var[:] = 1

            if hidden_sizes:
                hidden_layers = []
                hidden_layers.append(LinearBN(in_size, hidden_sizes[0]))
                for hin, hout in zip(hidden_sizes, hidden_sizes[1:]):
                    hidden_layers.append(LinearBN(hin, hout))
                self.hidden_layers = chainer.ChainList(*hidden_layers)
                self.output = L.Linear(hidden_sizes[-1], out_size,
                                       initialW=LeCunNormal(last_wscale))
            else:
                self.output = L.Linear(in_size, out_size,
                                       initialW=LeCunNormal(last_wscale))

            if normalize_output:
                self.output_bn = L.BatchNormalization(out_size)
                self.output_bn.avg_var[:] = 1

    def __call__(self, x):
        h = x
        assert (not chainer.config.train) or x.shape[0] > 1
        if self.normalize_input:
            h = self.input_bn(h)
        if self.hidden_sizes:
            for l in self.hidden_layers:
                h = self.nonlinearity(l(h))
        h = self.output(h)
        if self.normalize_output:
            h = self.output_bn(h)
        return h
