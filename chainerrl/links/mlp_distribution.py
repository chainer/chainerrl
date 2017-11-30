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
                self.output = L.Linear(hidden_sizes[-1], out_size * n_atoms,
                                       initialW=LeCunNormal(last_wscale))
            else:
                self.output = L.Linear(in_size, out_size * n_atoms,
                                       initialW=LeCunNormal(last_wscale))

    def __call__(self, x):
        h = x
        if self.hidden_sizes:
            for l in self.hidden_layers:
                h = self.nonlinearity(l(h))
        h = F.reshape(self.output(h), (-1, self.out_size, self.n_atoms))
        return F.softmax(h, axis=2)
