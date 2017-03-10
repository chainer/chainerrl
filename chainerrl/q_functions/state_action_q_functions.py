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

from chainerrl.links.mlp import MLP
from chainerrl.links.mlp_bn import MLPBN
from chainerrl.q_function import StateActionQFunction
from chainerrl.recurrent import RecurrentChainMixin


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


class FCSAQFunction(chainer.ChainList, StateActionQFunction):
    """Fully-connected (s,a)-input continuous Q-function.

    Args:
        n_dim_obs: number of dimensions of observation space
        n_dim_action: number of dimensions of action space
        n_hidden_channels: number of hidden channels
        n_hidden_layers: number of hidden layers
    """

    def __init__(self, n_dim_obs, n_dim_action, n_hidden_channels,
                 n_hidden_layers, nonlinearity=F.relu,
                 last_wscale=1):
        self.n_input_channels = n_dim_obs + n_dim_action
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels
        self.nonlinearity = nonlinearity

        layers = []
        assert self.n_hidden_layers >= 1
        layers.append(
            L.Linear(self.n_input_channels, self.n_hidden_channels))
        for i in range(self.n_hidden_layers - 1):
            layers.append(
                L.Linear(self.n_hidden_channels, self.n_hidden_channels))
        layers.append(L.Linear(self.n_hidden_channels, 1, wscale=last_wscale))
        super().__init__(*layers)
        self.output = layers[-1]

    def __call__(self, state, action, test=False):
        h = F.concat((state, action), axis=1)
        for layer in self[:-1]:
            h = self.nonlinearity(layer(h))
        h = self[-1](h)
        return h


class FCLSTMSAQFunction(chainer.Chain, StateActionQFunction,
                        RecurrentChainMixin):
    """Fully-connected (s,a)-input continuous Q-function.

    Args:
        n_dim_obs: number of dimensions of observation space
        n_dim_action: number of dimensions of action space
        n_hidden_channels: number of hidden channels
        n_hidden_layers: number of hidden layers
    """

    def __init__(self, n_dim_obs, n_dim_action, n_hidden_channels,
                 n_hidden_layers):
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
    """Fully-connected (s,a)-input continuous Q-function.

    Args:
        n_dim_obs: number of dimensions of observation space
        n_dim_action: number of dimensions of action space
        n_hidden_channels: number of hidden channels
        n_hidden_layers: number of hidden layers
    """

    def __init__(self, n_dim_obs, n_dim_action, n_hidden_channels,
                 n_hidden_layers, normalize_input=True):
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

    Args:
        n_dim_obs: number of dimensions of observation space
        n_dim_action: number of dimensions of action space
        n_hidden_channels: number of hidden channels
        n_hidden_layers: number of hidden layers
    """

    def __init__(self, n_dim_obs, n_dim_action, n_hidden_channels,
                 n_hidden_layers, normalize_input=True):
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
                    hidden_sizes=([self.n_hidden_channels] *
                                  (self.n_hidden_layers - 1))))
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

    Args:
        n_dim_obs: number of dimensions of observation space
        n_dim_action: number of dimensions of action space
        n_hidden_channels: number of hidden channels
        n_hidden_layers: number of hidden layers
    """

    def __init__(self, n_dim_obs, n_dim_action, n_hidden_channels,
                 n_hidden_layers):
        self.n_input_channels = n_dim_obs + n_dim_action
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels

        super().__init__(
            obs_mlp=MLP(in_size=n_dim_obs, out_size=n_hidden_channels,
                        hidden_sizes=[]),
            mlp=MLP(in_size=n_hidden_channels + n_dim_action,
                    out_size=1,
                    hidden_sizes=([self.n_hidden_channels] *
                                  (self.n_hidden_layers - 1))))
        self.output = self.mlp.output

    def __call__(self, state, action, test=False):
        h = F.relu(self.obs_mlp(state, test=test))
        h = F.concat((h, action), axis=1)
        return self.mlp(h, test=test)
