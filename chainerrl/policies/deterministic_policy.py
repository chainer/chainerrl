from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

from logging import getLogger
logger = getLogger(__name__)

import chainer
from chainer import functions as F
from chainer import links as L

from chainerrl import distribution
from chainerrl.functions.bound_by_tanh import bound_by_tanh
from chainerrl.links.mlp import MLP
from chainerrl.links.mlp_bn import MLPBN
from chainerrl.policy import Policy
from chainerrl.recurrent import RecurrentChainMixin


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

    def __init__(self, n_input_channels, n_hidden_layers,
                 n_hidden_channels, action_size,
                 min_action=None, max_action=None, bound_action=True):
        self.n_input_channels = n_input_channels
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels
        self.action_size = action_size
        self.min_action = min_action
        self.max_action = max_action
        self.bound_action = bound_action

        if self.bound_action:
            action_filter = lambda x: bound_by_tanh(
                x, self.min_action, self.max_action)
        else:
            action_filter = None

        super().__init__(
            model=MLP(n_input_channels,
                      action_size,
                      (n_hidden_channels,) * n_hidden_layers),
            action_filter=action_filter)


class FCBNDeterministicPolicy(ContinuousDeterministicPolicy):

    def __init__(self, n_input_channels, n_hidden_layers,
                 n_hidden_channels, action_size,
                 min_action=None, max_action=None, bound_action=True,
                 normalize_input=True):
        self.n_input_channels = n_input_channels
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels
        self.action_size = action_size
        self.min_action = min_action
        self.max_action = max_action
        self.bound_action = bound_action
        self.normalize_input = normalize_input

        if self.bound_action:
            action_filter = lambda x: bound_by_tanh(
                x, self.min_action, self.max_action)
        else:
            action_filter = None

        super().__init__(
            model=MLPBN(n_input_channels,
                        action_size,
                        (n_hidden_channels,) * n_hidden_layers,
                        normalize_input=self.normalize_input),
            action_filter=action_filter)


class FCLSTMDeterministicPolicy(ContinuousDeterministicPolicy):

    def __init__(self, n_input_channels, n_hidden_layers,
                 n_hidden_channels, action_size,
                 min_action=None, max_action=None, bound_action=True):
        self.n_input_channels = n_input_channels
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels
        self.action_size = action_size
        self.min_action = min_action
        self.max_action = max_action
        self.bound_action = bound_action

        if self.bound_action:
            action_filter = lambda x: bound_by_tanh(
                x, self.min_action, self.max_action)
        else:
            action_filter = None

        model = chainer.Chain(
            fc=MLP(self.n_input_channels,
                   n_hidden_channels,
                   (self.n_hidden_channels,) * self.n_hidden_layers),
            lstm=L.LSTM(n_hidden_channels, n_hidden_channels),
            out=L.Linear(n_hidden_channels, action_size),
        )

        def model_call(model, x):
            h = F.relu(model.fc(x))
            h = model.lstm(h)
            h = model.out(h)
            return h

        super().__init__(
            model=model,
            model_call=model_call,
            action_filter=action_filter)
