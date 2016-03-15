import random

import numpy as np
import chainer
from chainer import functions as F
from chainer import links as L
from chainer import cuda


class QFunction(object):
    pass


class StateInputQFunction(QFunction):
    """
    Input: state
    Output: values for each action
    """

    def forward(self, state, test=False):
        raise NotImplementedError()

    def sample_epsilon_greedily_with_value(self, state, epsilon):
        assert state.shape[0] == 1
        xp = cuda.get_array_module(state)
        values = self.forward(state)
        if random.random() < epsilon:
            a = random.randint(0, self.n_actions - 1)
        else:
            a = values.data[0].argmax()
        q = F.select_item(
            values, chainer.Variable(xp.asarray([a], dtype=np.int32)))
        return [a], q

    def sample_greedily_with_value(self, state):
        assert state.shape[0] == 1
        xp = cuda.get_array_module(state)
        values = self.forward(state)
        a = values.data[0].argmax()
        q = F.select_item(
            values, chainer.Variable(xp.asarray([a], dtype=np.int32)))
        return [a], q


class FCSIQFunction(chainer.ChainList, StateInputQFunction):

    def __init__(self, n_input_channels, n_actions, n_hidden_channels,
                 n_hidden_layers):
        self.n_input_channels = n_input_channels
        self.n_actions = n_actions
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels

        layers = []
        assert n_hidden_layers >= 1
        layers.append(L.Linear(n_input_channels, n_hidden_channels))
        for i in xrange(n_hidden_layers - 1):
            layers.append(L.Linear(n_hidden_channels, n_hidden_channels))
        layers.append(L.Linear(n_hidden_channels, n_actions))

        super(FCSIQFunction, self).__init__(*layers)

    def forward(self, state, test=False):
        h = chainer.Variable(state)
        for layer in self[:-1]:
            h = F.elu(layer(h))
        h = self[-1](h)
        return h
