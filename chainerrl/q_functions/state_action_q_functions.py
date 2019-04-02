from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import chainer
from chainer import functions as F
from chainer.initializers import LeCunNormal
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

    def __call__(self, x, a):
        h = self.model(x, a)
        return h


class FCSAQFunction(MLP, StateActionQFunction):
    """Fully-connected (s,a)-input Q-function.

    Args:
        n_dim_obs (int): Number of dimensions of observation space.
        n_dim_action (int): Number of dimensions of action space.
        n_hidden_channels (int): Number of hidden channels.
        n_hidden_layers (int): Number of hidden layers.
        nonlinearity (callable): Nonlinearity between layers. It must accept a
            Variable as an argument and return a Variable with the same shape.
            Nonlinearities with learnable parameters such as PReLU are not
            supported. It is not used if n_hidden_layers is zero.
        last_wscale (float): Scale of weight initialization of the last layer.
    """

    def __init__(self, n_dim_obs, n_dim_action, n_hidden_channels,
                 n_hidden_layers, nonlinearity=F.relu,
                 last_wscale=1.):
        self.n_input_channels = n_dim_obs + n_dim_action
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels
        self.nonlinearity = nonlinearity
        super().__init__(
            in_size=self.n_input_channels,
            out_size=1,
            hidden_sizes=[self.n_hidden_channels] * self.n_hidden_layers,
            nonlinearity=nonlinearity,
            last_wscale=last_wscale,
        )

    def __call__(self, state, action):
        h = F.concat((state, action), axis=1)
        return super().__call__(h)


class FCLSTMSAQFunction(chainer.Chain, StateActionQFunction,
                        RecurrentChainMixin):
    """Fully-connected + LSTM (s,a)-input Q-function.

    Args:
        n_dim_obs (int): Number of dimensions of observation space.
        n_dim_action (int): Number of dimensions of action space.
        n_hidden_channels (int): Number of hidden channels.
        n_hidden_layers (int): Number of hidden layers.
        nonlinearity (callable): Nonlinearity between layers. It must accept a
            Variable as an argument and return a Variable with the same shape.
            Nonlinearities with learnable parameters such as PReLU are not
            supported.
        last_wscale (float): Scale of weight initialization of the last layer.
    """

    def __init__(self, n_dim_obs, n_dim_action, n_hidden_channels,
                 n_hidden_layers, nonlinearity=F.relu, last_wscale=1.):
        self.n_input_channels = n_dim_obs + n_dim_action
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels
        self.nonlinearity = nonlinearity
        super().__init__()
        with self.init_scope():
            self.fc = MLP(self.n_input_channels, n_hidden_channels,
                          [self.n_hidden_channels] * self.n_hidden_layers,
                          nonlinearity=nonlinearity,
                          )
            self.lstm = L.LSTM(n_hidden_channels, n_hidden_channels)
            self.out = L.Linear(n_hidden_channels, 1,
                                initialW=LeCunNormal(last_wscale))

    def __call__(self, x, a):
        h = F.concat((x, a), axis=1)
        h = self.nonlinearity(self.fc(h))
        h = self.lstm(h)
        return self.out(h)


class FCBNSAQFunction(MLPBN, StateActionQFunction):
    """Fully-connected + BN (s,a)-input Q-function.

    Args:
        n_dim_obs (int): Number of dimensions of observation space.
        n_dim_action (int): Number of dimensions of action space.
        n_hidden_channels (int): Number of hidden channels.
        n_hidden_layers (int): Number of hidden layers.
        normalize_input (bool): If set to True, Batch Normalization is applied
            to both observations and actions.
        nonlinearity (callable): Nonlinearity between layers. It must accept a
            Variable as an argument and return a Variable with the same shape.
            Nonlinearities with learnable parameters such as PReLU are not
            supported. It is not used if n_hidden_layers is zero.
        last_wscale (float): Scale of weight initialization of the last layer.
    """

    def __init__(self, n_dim_obs, n_dim_action, n_hidden_channels,
                 n_hidden_layers, normalize_input=True,
                 nonlinearity=F.relu, last_wscale=1.):
        self.n_input_channels = n_dim_obs + n_dim_action
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels
        self.normalize_input = normalize_input
        self.nonlinearity = nonlinearity
        super().__init__(
            in_size=self.n_input_channels, out_size=1,
            hidden_sizes=[self.n_hidden_channels] * self.n_hidden_layers,
            normalize_input=self.normalize_input,
            nonlinearity=nonlinearity,
            last_wscale=last_wscale,
        )

    def __call__(self, state, action):
        h = F.concat((state, action), axis=1)
        return super().__call__(h)


class FCBNLateActionSAQFunction(chainer.Chain, StateActionQFunction):
    """Fully-connected + BN (s,a)-input Q-function with late action input.

    Actions are not included until the second hidden layer and not normalized.
    This architecture is used in the DDPG paper:
    http://arxiv.org/abs/1509.02971

    Args:
        n_dim_obs (int): Number of dimensions of observation space.
        n_dim_action (int): Number of dimensions of action space.
        n_hidden_channels (int): Number of hidden channels.
        n_hidden_layers (int): Number of hidden layers. It must be greater than
            or equal to 1.
        normalize_input (bool): If set to True, Batch Normalization is applied
        nonlinearity (callable): Nonlinearity between layers. It must accept a
            Variable as an argument and return a Variable with the same shape.
            Nonlinearities with learnable parameters such as PReLU are not
            supported.
        last_wscale (float): Scale of weight initialization of the last layer.
    """

    def __init__(self, n_dim_obs, n_dim_action, n_hidden_channels,
                 n_hidden_layers, normalize_input=True,
                 nonlinearity=F.relu, last_wscale=1.):
        assert n_hidden_layers >= 1
        self.n_input_channels = n_dim_obs + n_dim_action
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels
        self.normalize_input = normalize_input
        self.nonlinearity = nonlinearity

        super().__init__()
        with self.init_scope():
            # No need to pass nonlinearity to obs_mlp because it has no
            # hidden layers
            self.obs_mlp = MLPBN(in_size=n_dim_obs, out_size=n_hidden_channels,
                                 hidden_sizes=[],
                                 normalize_input=normalize_input,
                                 normalize_output=True)
            self.mlp = MLP(in_size=n_hidden_channels + n_dim_action,
                           out_size=1,
                           hidden_sizes=([self.n_hidden_channels] *
                                         (self.n_hidden_layers - 1)),
                           nonlinearity=nonlinearity,
                           last_wscale=last_wscale,
                           )

        self.output = self.mlp.output

    def __call__(self, state, action):
        h = self.nonlinearity(self.obs_mlp(state))
        h = F.concat((h, action), axis=1)
        return self.mlp(h)


class FCLateActionSAQFunction(chainer.Chain, StateActionQFunction):
    """Fully-connected (s,a)-input Q-function with late action input.

    Actions are not included until the second hidden layer and not normalized.
    This architecture is used in the DDPG paper:
    http://arxiv.org/abs/1509.02971

    Args:
        n_dim_obs (int): Number of dimensions of observation space.
        n_dim_action (int): Number of dimensions of action space.
        n_hidden_channels (int): Number of hidden channels.
        n_hidden_layers (int): Number of hidden layers. It must be greater than
            or equal to 1.
        nonlinearity (callable): Nonlinearity between layers. It must accept a
            Variable as an argument and return a Variable with the same shape.
            Nonlinearities with learnable parameters such as PReLU are not
            supported.
        last_wscale (float): Scale of weight initialization of the last layer.
    """

    def __init__(self, n_dim_obs, n_dim_action, n_hidden_channels,
                 n_hidden_layers, nonlinearity=F.relu, last_wscale=1.):
        assert n_hidden_layers >= 1
        self.n_input_channels = n_dim_obs + n_dim_action
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels
        self.nonlinearity = nonlinearity

        super().__init__()
        with self.init_scope():
            # No need to pass nonlinearity to obs_mlp because it has no
            # hidden layers
            self.obs_mlp = MLP(in_size=n_dim_obs, out_size=n_hidden_channels,
                               hidden_sizes=[])
            self.mlp = MLP(in_size=n_hidden_channels + n_dim_action,
                           out_size=1,
                           hidden_sizes=([self.n_hidden_channels] *
                                         (self.n_hidden_layers - 1)),
                           nonlinearity=nonlinearity,
                           last_wscale=last_wscale,
                           )

        self.output = self.mlp.output

    def __call__(self, state, action):
        h = self.nonlinearity(self.obs_mlp(state))
        h = F.concat((h, action), axis=1)
        return self.mlp(h)
