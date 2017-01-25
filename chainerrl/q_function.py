from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
from future.utils import with_metaclass
standard_library.install_aliases()

from abc import ABCMeta
from abc import abstractmethod

import numpy as np
import chainer
from chainer import functions as F
from chainer import links as L
from chainer import cuda

from chainerrl.links.mlp_bn import MLPBN
from chainerrl.links.mlp_wn import MLPWN
from chainerrl.links.mlp import MLP
from chainerrl.action_value import DiscreteActionValue
from chainerrl.action_value import QuadraticActionValue
from chainerrl.functions.lower_triangular_matrix import lower_triangular_matrix
from chainerrl.recurrent import RecurrentChainMixin


class StateQFunction(with_metaclass(ABCMeta, object)):

    @abstractmethod
    def __call__(self, x, test=False):
        raise NotImplementedError()


class StateActionQFunction(with_metaclass(ABCMeta, object)):

    @abstractmethod
    def __call__(self, x, a, test=False):
        raise NotImplementedError()


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


class SingleModelStateActionQFunction(
        chainer.Chain, StateActionQFunction, RecurrentChainMixin):
    """Q-function with discrete actions.

    Args:
        model (chainer.Link):
            Link that is callable and outputs action values.
    """

    def __init__(self, model):
        super().__init__(model=model)

    def __call__(self, x, a, test=False):
        h = self.model(x, a, test=test)
        return h


class FCStateQFunctionWithDiscreteAction(
        SingleModelStateQFunctionWithDiscreteAction):
    """Fully-connected state-input Q-function with discrete actions."""

    def __init__(self, ndim_obs, n_actions, n_hidden_channels,
                 n_hidden_layers):
        """
        Args:
          n_dim_obs: number of dimensions of observation space
          n_dim_action: number of dimensions of action space
          n_hidden_channels: number of hidden channels
          n_hidden_layers: number of hidden layers
        """
        super().__init__(model=MLP(
            in_size=ndim_obs, out_size=n_actions,
            hidden_sizes=[n_hidden_channels] * n_hidden_layers))


class FCLSTMStateQFunction(chainer.Chain, StateQFunction, RecurrentChainMixin):
    """Fully-connected state-input discrete  Q-function."""

    def __init__(self, n_dim_obs, n_dim_action, n_hidden_channels,
                 n_hidden_layers):
        """
        Args:
          n_dim_obs: number of dimensions of observation space
          n_dim_action: number of dimensions of action space
          n_hidden_channels: number of hidden channels before LSTM
          n_hidden_layers: number of hidden layers before LSTM
        """

        self.n_input_channels = n_dim_obs
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels
        self.state_stack = []
        super().__init__(
            fc=MLP(in_size=self.n_input_channels, out_size=n_hidden_channels,
                   hidden_sizes=[self.n_hidden_channels] * self.n_hidden_layers),
            lstm=L.LSTM(n_hidden_channels, n_hidden_channels),
            out=L.Linear(n_hidden_channels, n_dim_action)
        )

    def __call__(self, x, test=False):
        h = F.relu(self.fc(x, test=test))
        h = self.lstm(h)
        return DiscreteActionValue(self.out(h))


def scale_by_tanh(x, low, high):
    xp = cuda.get_array_module(x.data)
    scale = (high - low) / 2
    scale = xp.expand_dims(xp.asarray(scale, dtype=np.float32), axis=0)
    mean = (high + low) / 2
    mean = xp.expand_dims(xp.asarray(mean, dtype=np.float32), axis=0)
    return F.tanh(x) * scale + mean


class FCSIContinuousQFunction(chainer.Chain, StateQFunction):
    """Fully-connected state-input continuous Q-function."""

    def __init__(self, n_input_channels, n_dim_action, n_hidden_channels,
                 n_hidden_layers, action_space, scale_mu=True):
        """
        Args:
          n_input_channels: number of input channels
          n_dim_action: number of dimensions of action space
          n_hidden_channels: number of hidden channels
          n_hidden_layers: number of hidden layers
          action_space: action_space
          scale_mu (bool): scale mu by applying tanh if True
        """

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
        return QuadraticActionValue(mu, mat, v, min_action=self.action_space.low,
                                    max_action=self.action_space.high)


class FCBNSIContinuousQFunction(chainer.Chain, StateQFunction):
    """Fully-connected state-input continuous Q-function."""

    def __init__(self, n_input_channels, n_dim_action, n_hidden_channels,
                 n_hidden_layers, action_space, scale_mu=True,
                 normalize_input=True):
        """
        Args:
          n_input_channels: number of input channels
          n_dim_action: number of dimensions of action space
          n_hidden_channels: number of hidden channels
          n_hidden_layers: number of hidden layers
          action_space: action_space
          scale_mu (bool): scale mu by applying tanh if True
        """

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
        xp = cuda.get_array_module(state)
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
        return ContinuousQOutput(mu, mat, v, min_action=self.action_space.low,
                                 max_action=self.action_space.high)


class FCSAQFunction(chainer.ChainList, StateActionQFunction):
    """Fully-connected (s,a)-input continuous Q-function."""

    def __init__(self, n_dim_obs, n_dim_action, n_hidden_channels,
                 n_hidden_layers):
        """
        Args:
          n_dim_obs: number of dimensions of observation space
          n_dim_action: number of dimensions of action space
          n_hidden_channels: number of hidden channels
          n_hidden_layers: number of hidden layers
        """

        self.n_input_channels = n_dim_obs + n_dim_action
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels

        layers = []
        assert self.n_hidden_layers >= 1
        layers.append(
            L.Linear(self.n_input_channels, self.n_hidden_channels))
        for i in range(self.n_hidden_layers - 1):
            layers.append(
                L.Linear(self.n_hidden_channels, self.n_hidden_channels))
        layers.append(L.Linear(self.n_hidden_channels, 1))
        super().__init__(*layers)
        self.output = layers[-1]

    def __call__(self, state, action, test=False):
        h = F.concat((state, action), axis=1)
        for layer in self[:-1]:
            h = F.relu(layer(h))
        h = self[-1](h)
        return h


class FCLSTMSAQFunction(chainer.Chain, StateActionQFunction,
                        RecurrentChainMixin):
    """Fully-connected (s,a)-input continuous Q-function."""

    def __init__(self, n_dim_obs, n_dim_action, n_hidden_channels,
                 n_hidden_layers):
        """
        Args:
          n_dim_obs: number of dimensions of observation space
          n_dim_action: number of dimensions of action space
          n_hidden_channels: number of hidden channels
          n_hidden_layers: number of hidden layers
        """

        self.n_input_channels = n_dim_obs + n_dim_action
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels
        super().__init__(
            fc=MLP(self.n_input_channels, n_hidden_channels,
                   [self.n_hidden_channels] * self.n_hidden_layers),
            lstm=L.LSTM(n_hidden_channels, n_hidden_channels),
            out=L.Linear(n_hidden_channels, 1),
        )

    def __call__(self, x, a, test=False):
        h = F.concat((x, a), axis=1)
        h = F.relu(self.fc(h, test=test))
        h = self.lstm(h)
        return self.out(h)


class FCBNSAQFunction(MLPBN, StateActionQFunction):
    """Fully-connected (s,a)-input continuous Q-function."""

    def __init__(self, n_dim_obs, n_dim_action, n_hidden_channels,
                 n_hidden_layers, normalize_input=True):
        """
        Args:
          n_dim_obs: number of dimensions of observation space
          n_dim_action: number of dimensions of action space
          n_hidden_channels: number of hidden channels
          n_hidden_layers: number of hidden layers
        """

        self.n_input_channels = n_dim_obs + n_dim_action
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels
        self.normalize_input = normalize_input
        super().__init__(
            in_size=self.n_input_channels, out_size=1,
            hidden_sizes=[self.n_hidden_channels] * self.n_hidden_layers,
            normalize_input=self.normalize_input)

    def __call__(self, state, action, test=False):
        h = F.concat((state, action), axis=1)
        return super().__call__(h, test=test)


class FCBNLateActionSAQFunction(chainer.Chain, StateActionQFunction,
                                RecurrentChainMixin):
    """Fully-connected (s,a)-input continuous Q-function.

    Actions are not included until the second hidden layer and not normalized.
    This architecture is used in the DDPG paper:
    http://arxiv.org/abs/1509.02971
    """

    def __init__(self, n_dim_obs, n_dim_action, n_hidden_channels,
                 n_hidden_layers, normalize_input=True):
        """
        Args:
          n_dim_obs: number of dimensions of observation space
          n_dim_action: number of dimensions of action space
          n_hidden_channels: number of hidden channels
          n_hidden_layers: number of hidden layers
        """

        self.n_input_channels = n_dim_obs + n_dim_action
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels
        self.normalize_input = normalize_input

        super().__init__(
            obs_mlp=MLPBN(in_size=n_dim_obs, out_size=n_hidden_channels,
                          hidden_sizes=[], normalize_input=normalize_input,
                          normalize_output=True),
            mlp=MLP(in_size=n_hidden_channels + n_dim_action,
                    out_size=1,
                    hidden_sizes=[self.n_input_channels] * (self.n_hidden_layers - 1)))
        self.output = self.mlp.output

    def __call__(self, state, action, test=False):
        h = F.relu(self.obs_mlp(state, test=test))
        h = F.concat((h, action), axis=1)
        return self.mlp(h, test=test)


class FCLateActionSAQFunction(chainer.Chain, StateActionQFunction,
                              RecurrentChainMixin):
    """Fully-connected (s,a)-input continuous Q-function.

    Actions are not included until the second hidden layer and not normalized.
    This architecture is used in the DDPG paper:
    http://arxiv.org/abs/1509.02971
    """

    def __init__(self, n_dim_obs, n_dim_action, n_hidden_channels,
                 n_hidden_layers):
        """
        Args:
          n_dim_obs: number of dimensions of observation space
          n_dim_action: number of dimensions of action space
          n_hidden_channels: number of hidden channels
          n_hidden_layers: number of hidden layers
        """

        self.n_input_channels = n_dim_obs + n_dim_action
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels

        super().__init__(
            obs_mlp=MLP(in_size=n_dim_obs, out_size=n_hidden_channels,
                        hidden_sizes=[]),
            mlp=MLP(in_size=n_hidden_channels + n_dim_action,
                    out_size=1,
                    hidden_sizes=[self.n_input_channels] * (self.n_hidden_layers - 1)))
        self.output = self.mlp.output

    def __call__(self, state, action, test=False):
        h = F.relu(self.obs_mlp(state, test=test))
        h = F.concat((h, action), axis=1)
        return self.mlp(h, test=test)
