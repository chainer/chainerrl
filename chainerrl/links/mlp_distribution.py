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
from chainer import cuda

from chainerrl.initializers import LeCunNormal


class MLPDistribution(chainer.Chain):
    """Multi-Layer Perceptron"""

    def __init__(self, in_size, n_atoms, out_size, hidden_sizes, nonlinearity=F.relu,
                 last_wscale=1):
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_sizes = hidden_sizes
        self.nonlinearity = nonlinearity
        self.n_atoms = n_atoms

        super().__init__()
        with self.init_scope():
            if hidden_sizes:
                hidden_layers = []
                hidden_layers.append(L.Linear(in_size, hidden_sizes[0]))
                for hin, hout in zip(hidden_sizes, hidden_sizes[1:]):
                    hidden_layers.append(L.Linear(hin, hout))
                self.hidden_layers = chainer.ChainList(*hidden_layers)
                self.outputs = chainer.ChainList(*[L.Linear(hidden_sizes[-1], n_atoms,
                                       initialW=LeCunNormal(last_wscale)) for _ in range(out_size)])
            else:
                self.outputs = chainer.ChainList(*[L.Linear(in_size, out_size, n_atoms,
                                       initialW=LeCunNormal(last_wscale)) for _ in range(out_size)])

    def __call__(self, x):
        h = x# / 255.0
        #print("input:", h)
        if self.hidden_sizes:
            for i, l in enumerate(self.hidden_layers):
                h = self.nonlinearity(l(h))
                #print("hidden:", h)

        outs = []
        for output in self.outputs:
            #print("preout hidden:", h)
            a = F.softmax(output(h), axis=1)
            #print("out:", a)
            outs.append(a)

        h = F.stack(outs, axis=1)

        #print("model out:", h.shape)

        return h
