from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import chainer
from chainer import functions as F
from chainer import links as L
import numpy as np

from chainerrl import action_value
from chainerrl.links.mlp import MLP
from chainerrl.q_function import StateQFunction
from chainerrl.recurrent import RecurrentChainMixin


class DuelingDQN(chainer.Chain, StateQFunction):
    """Dueling Q-Network

    See: http://arxiv.org/abs/1511.06581
    """

    def __init__(self, n_actions, n_input_channels=4,
                 activation=F.relu, bias=0.1):
        self.n_actions = n_actions
        self.n_input_channels = n_input_channels
        self.activation = activation

        super().__init__()
        with self.init_scope():
            self.conv_layers = chainer.ChainList(
                L.Convolution2D(n_input_channels, 32, 8, stride=4,
                                initial_bias=bias),
                L.Convolution2D(32, 64, 4, stride=2, initial_bias=bias),
                L.Convolution2D(64, 64, 3, stride=1, initial_bias=bias))

            self.a_stream = MLP(3136, n_actions, [512])
            self.v_stream = MLP(3136, 1, [512])

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


class DistributionalDuelingDQN(
        chainer.Chain, StateQFunction, RecurrentChainMixin):
    """Distributional dueling fully-connected Q-function with discrete actions.

    """

    def __init__(self, n_actions, n_atoms, v_min, v_max,
                 n_input_channels=4, activation=F.relu, bias=0.1):
        assert n_atoms >= 2
        assert v_min < v_max

        self.n_actions = n_actions
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.n_atoms = n_atoms

        super().__init__()
        z_values = self.xp.linspace(v_min, v_max,
                                    num=n_atoms,
                                    dtype=np.float32)
        self.add_persistent('z_values', z_values)

        with self.init_scope():
            self.conv_layers = chainer.ChainList(
                L.Convolution2D(n_input_channels, 32, 8, stride=4,
                                initial_bias=bias),
                L.Convolution2D(32, 64, 4, stride=2, initial_bias=bias),
                L.Convolution2D(64, 64, 3, stride=1, initial_bias=bias))

            self.main_stream = L.Linear(3136, 1024)
            self.a_stream = L.Linear(512, n_actions * n_atoms)
            self.v_stream = L.Linear(512, n_atoms)

    def __call__(self, x):
        h = x
        for l in self.conv_layers:
            h = self.activation(l(h))

        # Advantage
        batch_size = x.shape[0]

        h = self.activation(self.main_stream(h))
        h_a, h_v = F.split_axis(h, 2, axis=-1)
        ya = F.reshape(self.a_stream(h_a),
                       (batch_size, self.n_actions, self.n_atoms))

        mean = F.sum(ya, axis=1, keepdims=True) / self.n_actions

        ya, mean = F.broadcast(ya, mean)
        ya -= mean

        # State value
        ys = F.reshape(self.v_stream(h_v), (batch_size, 1, self.n_atoms))
        ya, ys = F.broadcast(ya, ys)
        q = F.softmax(ya + ys, axis=2)

        return action_value.DistributionalDiscreteActionValue(q, self.z_values)
