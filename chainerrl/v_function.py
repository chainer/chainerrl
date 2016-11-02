from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import super
from builtins import range
from future import standard_library
standard_library.install_aliases()

import chainer
from chainer import functions as F
from chainer import links as L

from chainerrl import stateful_callable


class VFunction(stateful_callable.StatefulCallable):

    def push_state(self):
        pass

    def pop_state(self):
        pass

    def reset_state(self):
        pass

    def push_and_keep_state(self):
        pass

    def update_state(self, x, test=False):
        """Update its state so that it reflects x and a.

        Unlike __call__, stateless QFunctions would do nothing.
        """
        pass


class FCVFunction(chainer.ChainList, VFunction):

    def __init__(self, n_input_channels, n_hidden_layers=0,
                 n_hidden_channels=None):
        self.n_input_channels = n_input_channels
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels

        layers = []
        if n_hidden_layers > 0:
            layers.append(L.Linear(n_input_channels, n_hidden_channels))
            for i in range(n_hidden_layers - 1):
                layers.append(L.Linear(n_hidden_channels, n_hidden_channels))
            layers.append(L.Linear(n_hidden_channels, 1))
        else:
            layers.append(L.Linear(n_input_channels, 1))

        super(FCVFunction, self).__init__(*layers)

    def __call__(self, state):
        h = state
        for layer in self[:-1]:
            h = F.relu(layer(h))
        h = self[-1](h)
        return h
