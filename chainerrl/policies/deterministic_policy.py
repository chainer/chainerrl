from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import super
from builtins import range
from future import standard_library
from future.utils import with_metaclass
standard_library.install_aliases()

from logging import getLogger
logger = getLogger(__name__)

from abc import ABCMeta
from abc import abstractmethod

import chainer
from chainer import cuda
from chainer import functions as F
from chainer import links as L

from chainerrl.links.mlp_bn import MLPBN
from chainerrl.links.mlp import MLP
from chainerrl import distribution
from chainerrl import policy
from chainerrl.functions.bound_by_tanh import bound_by_tanh


class ContinuousDeterministicPolicy(policy.Policy):
    """Abstract deterministic policy."""

    @abstractmethod
    def compute_action(self, x, test=False):
        """
        Returns:
          ~chainer.Variable: action
        """
        raise NotImplementedError()

    def __call__(self, x, test=False):
        action = self.compute_action(x, test=test)
        return distribution.ContinuousDeterministicDistribution(action)


class FCDeterministicPolicy(chainer.ChainList, ContinuousDeterministicPolicy):

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

        layers = []
        if n_hidden_layers > 0:
            layers.append(L.Linear(n_input_channels, n_hidden_channels))
            for i in range(n_hidden_layers - 1):
                layers.append(L.Linear(n_hidden_channels, n_hidden_channels))
            layers.append(L.Linear(n_hidden_channels, self.action_size))
        else:
            layers.append(L.Linear(n_input_channels, self.action_size))

        super().__init__(*layers)

    def compute_action(self, state, test=False):
        h = state
        for layer in self[:-1]:
            h = F.relu(layer(h))
        a = self[-1](h)

        if self.bound_action:
            a = bound_by_tanh(a, self.min_action, self.max_action)

        return a


class FCLSTMDeterministicPolicy(chainer.Chain, ContinuousDeterministicPolicy):

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

        self.state_stack = []
        super().__init__(
            fc=MLP(in_size=self.n_input_channels, out_size=n_hidden_channels,
                   hidden_sizes=[self.n_hidden_channels] * self.n_hidden_layers),
            lstm=L.LSTM(n_hidden_channels, n_hidden_channels),
            out=L.Linear(n_hidden_channels, self.action_size)
        )

    def compute_action(self, x, test=False):
        h = F.relu(self.fc(x, test=test))
        h = F.relu(self.lstm(h))
        a = self.out(h)
        if self.bound_action:
            a = bound_by_tanh(a, self.min_action, self.max_action)
        return a

    def push_state(self):
        self.state_stack.append((self.lstm.h, self.lstm.c))
        self.lstm.reset_state()

    def pop_state(self):
        h, c = self.state_stack.pop()
        if h is not None and c is not None:
            self.lstm.set_state(c=c, h=h)

    def reset_state(self):
        self.lstm.reset_state()

    def update_state(self, x, test=False):
        self.__call__(x, test=test)


class FCBNDeterministicPolicy(chainer.Chain, ContinuousDeterministicPolicy):

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

        super().__init__(mlp=MLPBN(
            in_size=self.n_input_channels,
            out_size=self.action_size,
            hidden_sizes=[self.n_hidden_channels] * self.n_hidden_layers,
            normalize_input=self.normalize_input))

    def compute_action(self, state, test=False):
        a = self.mlp(state, test=test)

        if self.bound_action:
            a = bound_by_tanh(a, self.min_action, self.max_action)

        return a
