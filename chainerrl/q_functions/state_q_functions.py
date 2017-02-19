from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

import chainer
from chainer import cuda
from chainer import functions as F
from chainer import links as L
import numpy as np

from chainerrl.action_value import DiscreteActionValue
from chainerrl.action_value import QuadraticActionValue
from chainerrl.functions.lower_triangular_matrix import lower_triangular_matrix
from chainerrl.links.mlp import MLP
from chainerrl.links.mlp_bn import MLPBN
from chainerrl.q_function import StateQFunction
from chainerrl.recurrent import RecurrentChainMixin


def scale_by_tanh(x, low, high):
    xp = cuda.get_array_module(x.data)
    scale = (high - low) / 2
    scale = xp.expand_dims(xp.asarray(scale, dtype=np.float32), axis=0)
    mean = (high + low) / 2
    mean = xp.expand_dims(xp.asarray(mean, dtype=np.float32), axis=0)
    return F.tanh(x) * scale + mean


class SingleModelStateQFunctionWithDiscreteAction(
        chainer.Chain, StateQFunction, RecurrentChainMixin):
    """Q-function with discrete actions.

    Args:
        model (chainer.Link):
            Link that is callable and outputs action values.
    """

    def __init__(self, model):
        super().__init__(model=model)

    def __call__(self, x, test=False):
        h = self.model(x, test=test)
        return DiscreteActionValue(h)


class FCStateQFunctionWithDiscreteAction(
        SingleModelStateQFunctionWithDiscreteAction):
    """Fully-connected state-input Q-function with discrete actions.

    Args:
        n_dim_obs: number of dimensions of observation space
        n_dim_action: number of dimensions of action space
        n_hidden_channels: number of hidden channels
        n_hidden_layers: number of hidden layers
    """

    def __init__(self, ndim_obs, n_actions, n_hidden_channels,
                 n_hidden_layers):
        super().__init__(model=MLP(
            in_size=ndim_obs, out_size=n_actions,
            hidden_sizes=[n_hidden_channels] * n_hidden_layers))


class FCLSTMStateQFunction(chainer.Chain, StateQFunction, RecurrentChainMixin):
    """Fully-connected state-input discrete  Q-function.

    Args:
        n_dim_obs: number of dimensions of observation space
        n_dim_action: number of dimensions of action space
        n_hidden_channels: number of hidden channels before LSTM
        n_hidden_layers: number of hidden layers before LSTM
    """

    def __init__(self, n_dim_obs, n_dim_action, n_hidden_channels,
                 n_hidden_layers):
        self.n_input_channels = n_dim_obs
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels
        self.state_stack = []
        super().__init__(
            fc=MLP(in_size=self.n_input_channels, out_size=n_hidden_channels,
                   hidden_sizes=[self.n_hidden_channels] *
                   self.n_hidden_layers),
            lstm=L.LSTM(n_hidden_channels, n_hidden_channels),
            out=L.Linear(n_hidden_channels, n_dim_action)
        )

    def __call__(self, x, test=False):
        h = F.relu(self.fc(x, test=test))
        h = self.lstm(h)
        return DiscreteActionValue(self.out(h))


class FCQuadraticStateQFunction(
        chainer.Chain, StateQFunction):
    """Fully-connected state-input continuous Q-function.

    Args:
        n_input_channels: number of input channels
        n_dim_action: number of dimensions of action space
        n_hidden_channels: number of hidden channels
        n_hidden_layers: number of hidden layers
        action_space: action_space
        scale_mu (bool): scale mu by applying tanh if True
    """

    def __init__(self, n_input_channels, n_dim_action, n_hidden_channels,
                 n_hidden_layers, action_space, scale_mu=True):
        self.n_input_channels = n_input_channels
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels

        assert action_space is not None
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
        h = state
        for layer in self.hidden_layers:
            h = F.relu(layer(h))
        v = self.v(h)
        mu = self.mu(h)

        if self.scale_mu:
            mu = scale_by_tanh(mu, high=self.action_space.high,
                               low=self.action_space.low)

        mat_diag = F.exp(self.mat_diag(h))
        if hasattr(self, 'mat_non_diag'):
            mat_non_diag = self.mat_non_diag(h)
            tril = lower_triangular_matrix(mat_diag, mat_non_diag)
            mat = F.batch_matmul(tril, tril, transb=True)
        else:
            mat = F.expand_dims(mat_diag ** 2, axis=2)
        return QuadraticActionValue(
            mu, mat, v, min_action=self.action_space.low,
            max_action=self.action_space.high)


class FCBNQuadraticStateQFunction(chainer.Chain, StateQFunction):
    """Fully-connected state-input continuous Q-function.

    Args:
        n_input_channels: number of input channels
        n_dim_action: number of dimensions of action space
        n_hidden_channels: number of hidden channels
        n_hidden_layers: number of hidden layers
        action_space: action_space
        scale_mu (bool): scale mu by applying tanh if True
    """

    def __init__(self, n_input_channels, n_dim_action, n_hidden_channels,
                 n_hidden_layers, action_space, scale_mu=True,
                 normalize_input=True):
        self.n_input_channels = n_input_channels
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels

        assert action_space is not None
        self.scale_mu = scale_mu
        self.action_space = action_space

        layers = {}

        assert n_hidden_layers >= 1
        layers['hidden_layers'] = MLPBN(
            in_size=n_input_channels, out_size=n_hidden_channels,
            hidden_sizes=[n_hidden_channels] * (n_hidden_layers - 1),
            normalize_input=normalize_input)

        layers['v'] = L.Linear(n_hidden_channels, 1)
        layers['mu'] = L.Linear(n_hidden_channels, n_dim_action)
        layers['mat_diag'] = L.Linear(n_hidden_channels, n_dim_action)
        non_diag_size = n_dim_action * (n_dim_action - 1) // 2
        if non_diag_size > 0:
            layers['mat_non_diag'] = L.Linear(n_hidden_channels, non_diag_size)

        super().__init__(**layers)

    def __call__(self, state, test=False):
        h = self.hidden_layers(state, test=test)
        v = self.v(h)
        mu = self.mu(h)

        if self.scale_mu:
            mu = scale_by_tanh(mu, high=self.action_space.high,
                               low=self.action_space.low)

        mat_diag = F.exp(self.mat_diag(h))
        if hasattr(self, 'mat_non_diag'):
            mat_non_diag = self.mat_non_diag(h)
            tril = lower_triangular_matrix(mat_diag, mat_non_diag)
            mat = F.batch_matmul(tril, tril, transb=True)
        else:
            mat = F.expand_dims(mat_diag ** 2, axis=2)
        return QuadraticActionValue(
            mu, mat, v, min_action=self.action_space.low,
            max_action=self.action_space.high)
