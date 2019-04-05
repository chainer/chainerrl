from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

from logging import getLogger

import chainer
from chainer import functions as F
from chainer.initializers import LeCunNormal
from chainer import links as L

from chainerrl import distribution
from chainerrl.functions.bound_by_tanh import bound_by_tanh
from chainerrl.links.mlp import MLP
from chainerrl.links.mlp_bn import MLPBN
from chainerrl.policy import Policy
from chainerrl.recurrent import RecurrentChainMixin


logger = getLogger(__name__)


class ContinuousDeterministicPolicy(
        chainer.Chain, Policy, RecurrentChainMixin):
    """Continuous deterministic policy.

    Args:
        model (chainer.Link):
            Link that is callable and outputs action values.
        model_call (callable or None):
            Callable used instead of model.__call__ if not None
        action_filter (callable or None):
            Callable applied to the outputs of the model if not None
    """

    def __init__(self, model, model_call=None, action_filter=None):
        super().__init__(model=model)
        self.model_call = model_call
        self.action_filter = action_filter

    def __call__(self, x):
        # Model
        if self.model_call is not None:
            h = self.model_call(self.model, x)
        else:
            h = self.model(x)
        # Action filter
        if self.action_filter is not None:
            h = self.action_filter(h)
        # Wrap by Distribution
        return distribution.ContinuousDeterministicDistribution(h)


class FCDeterministicPolicy(ContinuousDeterministicPolicy):
    """Fully-connected deterministic policy.

    Args:
        n_input_channels (int): Number of input channels.
        n_hidden_layers (int): Number of hidden layers.
        n_hidden_channels (int): Number of hidden channels.
        action_size (int): Size of actions.
        min_action (ndarray or None): Minimum action. Used only if bound_action
            is set to True.
        min_action (ndarray or None): Minimum action. Used only if bound_action
            is set to True.
        bound_action (bool): If set to True, actions are bounded to
            [min_action, max_action] by tanh.
        nonlinearity (callable): Nonlinearity between layers. It must accept a
            Variable as an argument and return a Variable with the same shape.
            Nonlinearities with learnable parameters such as PReLU are not
            supported. It is not used if n_hidden_layers is zero.
        last_wscale (float): Scale of weight initialization of the last layer.
    """

    def __init__(self, n_input_channels, n_hidden_layers,
                 n_hidden_channels, action_size,
                 min_action=None, max_action=None, bound_action=True,
                 nonlinearity=F.relu,
                 last_wscale=1.):
        self.n_input_channels = n_input_channels
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels
        self.action_size = action_size
        self.min_action = min_action
        self.max_action = max_action
        self.bound_action = bound_action

        if self.bound_action:
            def action_filter(x):
                return bound_by_tanh(
                    x, self.min_action, self.max_action)
        else:
            action_filter = None

        super().__init__(
            model=MLP(n_input_channels,
                      action_size,
                      (n_hidden_channels,) * n_hidden_layers,
                      nonlinearity=nonlinearity,
                      last_wscale=last_wscale,
                      ),
            action_filter=action_filter)


class FCBNDeterministicPolicy(ContinuousDeterministicPolicy):
    """Fully-connected deterministic policy with Batch Normalization.

    Args:
        n_input_channels (int): Number of input channels.
        n_hidden_layers (int): Number of hidden layers.
        n_hidden_channels (int): Number of hidden channels.
        action_size (int): Size of actions.
        min_action (ndarray or None): Minimum action. Used only if bound_action
            is set to True.
        min_action (ndarray or None): Minimum action. Used only if bound_action
            is set to True.
        bound_action (bool): If set to True, actions are bounded to
            [min_action, max_action] by tanh.
        normalize_input (bool): If set to True, Batch Normalization is applied
            to inputs as well as hidden activations.
        nonlinearity (callable): Nonlinearity between layers. It must accept a
            Variable as an argument and return a Variable with the same shape.
            Nonlinearities with learnable parameters such as PReLU are not
            supported. It is not used if n_hidden_layers is zero.
        last_wscale (float): Scale of weight initialization of the last layer.
    """

    def __init__(self, n_input_channels, n_hidden_layers,
                 n_hidden_channels, action_size,
                 min_action=None, max_action=None, bound_action=True,
                 normalize_input=True,
                 nonlinearity=F.relu,
                 last_wscale=1.):
        self.n_input_channels = n_input_channels
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels
        self.action_size = action_size
        self.min_action = min_action
        self.max_action = max_action
        self.bound_action = bound_action
        self.normalize_input = normalize_input

        if self.bound_action:
            def action_filter(x):
                return bound_by_tanh(
                    x, self.min_action, self.max_action)
        else:
            action_filter = None

        super().__init__(
            model=MLPBN(n_input_channels,
                        action_size,
                        (n_hidden_channels,) * n_hidden_layers,
                        normalize_input=self.normalize_input,
                        nonlinearity=nonlinearity,
                        last_wscale=last_wscale,
                        ),
            action_filter=action_filter)


class FCLSTMDeterministicPolicy(ContinuousDeterministicPolicy):
    """Fully-connected deterministic policy with LSTM.

    Args:
        n_input_channels (int): Number of input channels.
        n_hidden_layers (int): Number of hidden layers.
        n_hidden_channels (int): Number of hidden channels.
        action_size (int): Size of actions.
        min_action (ndarray or None): Minimum action. Used only if bound_action
            is set to True.
        min_action (ndarray or None): Minimum action. Used only if bound_action
            is set to True.
        bound_action (bool): If set to True, actions are bounded to
            [min_action, max_action] by tanh.
        nonlinearity (callable): Nonlinearity between layers. It must accept a
            Variable as an argument and return a Variable with the same shape.
            Nonlinearities with learnable parameters such as PReLU are not
            supported.
        last_wscale (float): Scale of weight initialization of the last layer.
        """

    def __init__(self, n_input_channels, n_hidden_layers,
                 n_hidden_channels, action_size,
                 min_action=None, max_action=None, bound_action=True,
                 nonlinearity=F.relu,
                 last_wscale=1.):
        self.n_input_channels = n_input_channels
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels
        self.action_size = action_size
        self.min_action = min_action
        self.max_action = max_action
        self.bound_action = bound_action

        if self.bound_action:
            def action_filter(x):
                return bound_by_tanh(
                    x, self.min_action, self.max_action)
        else:
            action_filter = None

        model = chainer.Chain(
            fc=MLP(self.n_input_channels,
                   n_hidden_channels,
                   (self.n_hidden_channels,) * self.n_hidden_layers,
                   nonlinearity=nonlinearity,
                   ),
            lstm=L.LSTM(n_hidden_channels, n_hidden_channels),
            out=L.Linear(n_hidden_channels, action_size,
                         initialW=LeCunNormal(last_wscale)),
        )

        def model_call(model, x):
            h = nonlinearity(model.fc(x))
            h = model.lstm(h)
            h = model.out(h)
            return h

        super().__init__(
            model=model,
            model_call=model_call,
            action_filter=action_filter)
