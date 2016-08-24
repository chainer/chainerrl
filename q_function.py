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


class QFunction(object):
    pass


class StateInputQFunction(QFunction):
    """
    Input: state
    Output: values for each action
    """

    def forward(self, state, test=False):
        raise NotImplementedError()

    def __call__(self, state, action):
        assert isinstance(state, chainer.Variable)
        assert state.data.shape[0] == 1
        xp = cuda.get_array_module(state.data)
        values = self.forward(state)
        q = F.select_item(
            values, chainer.Variable(xp.asarray(action, dtype=np.int32)))
        return q

    def sample_epsilon_greedily_with_value(self, state, epsilon):
        assert isinstance(state, chainer.Variable)
        assert state.data.shape[0] == 1
        xp = cuda.get_array_module(state.data)
        values = self.forward(state)
        if random.random() < epsilon:
            a = random.randint(0, self.n_actions - 1)
        else:
            a = values.data[0].argmax()
        q = F.select_item(
            values, chainer.Variable(xp.asarray([a], dtype=np.int32)))
        return [a], q

    def sample_greedily_with_value(self, state):
        assert isinstance(state, chainer.Variable)
        assert state.data.shape[0] == 1
        xp = cuda.get_array_module(state.data)
        values = self.forward(state)
        a = values.data[0].argmax()
        q = F.select_item(
            values, chainer.Variable(xp.asarray([a], dtype=np.int32)))
        return [a], q


class FCSIQFunction(chainer.ChainList, QFunction):
    """Fully-connected state-input (discrete) Q-function."""

    def __init__(self, n_input_channels, n_actions, n_hidden_channels,
                 n_hidden_layers):
        self.n_input_channels = n_input_channels
        self.n_actions = n_actions
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels

        layers = []
        assert n_hidden_layers >= 1
        layers.append(L.Linear(n_input_channels, n_hidden_channels))
        for i in range(n_hidden_layers - 1):
            layers.append(L.Linear(n_hidden_channels, n_hidden_channels))
        layers.append(L.Linear(n_hidden_channels, n_actions))

        super(FCSIQFunction, self).__init__(*layers)

    def __call__(self, state, test=False):
        assert isinstance(state, chainer.Variable)
        h = state
        for layer in self[:-1]:
            h = F.elu(layer(h))
        h = self[-1](h)
        return DiscreteQOutput(h)


class FCSIContinuousQFunction(chainer.Chain, QFunction):
    """Fully-connected state-input continuous Q-function."""

    def __init__(self, n_input_channels, n_dim_action, n_hidden_channels,
                 n_hidden_layers, action_space=None, scale_mu=True):
        """
        Args:
          n_input_channels: number of input channels
          n_dim_action: number of dimensions of action space
          n_hidden_channels: number of hidden channels
          n_hidden_layers: number of hidden layers
          scale_mu (bool): scale mu by applying tanh if True
          action_space: action_space, only necessary if scale_mu=True
        """

        self.n_input_channels = n_input_channels
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels

        assert (not scale_mu) or (action_space is not None)
        self.scale_mu = scale_mu
        self.action_space = action_space

        layers = {}

        hidden_layers = []
        assert n_hidden_layers >= 1
        hidden_layers.append(L.Linear(n_input_channels, n_hidden_channels))
        for i in range(n_hidden_layers - 1):
            hidden_layers.append(
                L.Linear(n_hidden_channels, n_hidden_channels))
        layers['hidden_layers'] = chainer.ChainList(*hidden_layers)

        layers['v'] = L.Linear(n_hidden_channels, 1)
        layers['mu'] = L.Linear(n_hidden_channels, n_dim_action)
        layers['mat_diag'] = L.Linear(n_hidden_channels, n_dim_action)
        non_diag_size = n_dim_action * (n_dim_action - 1) // 2
        if non_diag_size > 0:
            layers['mat_non_diag'] = L.Linear(n_hidden_channels, non_diag_size)

        super().__init__(**layers)

    def __call__(self, state, test=False):
        assert isinstance(state, chainer.Variable)
        xp = cuda.get_array_module(state.data)
        h = state
        for layer in self.hidden_layers:
            h = F.elu(layer(h))
        v = self.v(h)
        mu = self.mu(h)

        if self.scale_mu:
            action_scale = (self.action_space.high - self.action_space.low) / 2
            action_scale = xp.expand_dims(xp.asarray(action_scale), axis=0)
            action_mean = (self.action_space.high + self.action_space.low) / 2
            action_mean = xp.expand_dims(xp.asarray(action_mean), axis=0)
            mu = F.tanh(mu) * action_scale + action_mean

        mat_diag = F.exp(self.mat_diag(h))
        if hasattr(self, 'mat_non_diag'):
            mat_non_diag = self.mat_non_diag(h)
            tril = lower_triangular_matrix(mat_diag, mat_non_diag)
            mat = F.batch_matmul(tril, tril, transb=True)
        else:
            mat = F.expand_dims(mat_diag ** 2, axis=2)
        return ContinuousQOutput(mu, mat, v)
