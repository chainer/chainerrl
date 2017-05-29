from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

import chainer
from chainer import functions as F
from chainer import links as L

from chainerrl import action_value
from chainerrl.links.mlp import MLP
from chainerrl.q_function import StateQFunction


class DuelingDQN(chainer.Chain, StateQFunction):

    def __init__(self, n_actions, n_input_channels=4,
                 activation=F.relu, bias=0.1):
        self.n_actions = n_actions
        self.n_input_channels = n_input_channels
        self.activation = activation

        conv_layers = chainer.ChainList(
            L.Convolution2D(n_input_channels, 32, 8, stride=4,
                            initial_bias=bias),
            L.Convolution2D(32, 64, 4, stride=2, initial_bias=bias),
            L.Convolution2D(64, 64, 3, stride=1, initial_bias=bias))

        a_stream = MLP(3136, n_actions, [512])
        v_stream = MLP(3136, 1, [512])

        super().__init__(conv_layers=conv_layers,
                         a_stream=a_stream,
                         v_stream=v_stream)

    def __call__(self, x):
        h = x
        for l in self.conv_layers:
            h = self.activation(l(h))

        # Advantage
        batch_size = x.shape[0]
        ya = self.a_stream(h)
        mean = F.reshape(
            F.sum(ya, axis=1) / self.n_actions, (batch_size, 1))
        ya, mean = F.broadcast(ya, mean)
        ya -= mean

        # State value
        ys = self.v_stream(h)

        ya, ys = F.broadcast(ya, ys)
        q = ya + ys
        return action_value.DiscreteActionValue(q)
