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

import policy_output
from links.mlp_bn import MLPBN
from links.mlp import MLP
from chainerrl import distribution


class Policy(with_metaclass(ABCMeta, object)):
    """Abstract policy."""

    @abstractmethod
    def __call__(self, state, test=False):
        """
        Returns:
            Distribution of actions
        """
        raise NotImplementedError()

    def push_state(self):
        pass

    def pop_state(self):
        pass

    def reset_state(self):
        pass

    def update_state(self, x, test=False):
        """Update its state so that it reflects x and a.

        Unlike __call__, stateless QFunctions would do nothing.
        """
        pass


class SoftmaxPolicy(Policy):
    """Abstract softmax policy."""

    @abstractmethod
    def compute_logits(self, state):
        """
        Returns:
          ~chainer.Variable: logits of actions
        """
        raise NotImplementedError()

    def __call__(self, state):
        return distribution.SoftmaxDistribution(self.compute_logits(state))


class GaussianPolicy(Policy):
    """Abstract Gaussian policy."""

    @abstractmethod
    def compute_mean_and_var(self, x, test=False):
        """
        Returns:
          tuple of two ~chainer.Variable: mean and variance
        """
        raise NotImplementedError()

    def __call__(self, x, test=False):
        mean, var = self.compute_mean_and_var(x, test=test)
        return distribution.GaussianDistribution(mean=mean, var=var)


class ContinuousDeterministicPolicy(Policy):
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


class FCSoftmaxPolicy(chainer.ChainList, SoftmaxPolicy):
    """Softmax policy that consists of FC layers and rectifiers"""

    def __init__(self, n_input_channels, n_actions,
                 n_hidden_layers=0, n_hidden_channels=None):
        self.n_input_channels = n_input_channels
        self.n_actions = n_actions
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels

        layers = []
        if n_hidden_layers > 0:
            layers.append(L.Linear(n_input_channels, n_hidden_channels))
            for i in range(n_hidden_layers - 1):
                layers.append(L.Linear(n_hidden_channels, n_hidden_channels))
            layers.append(L.Linear(n_hidden_channels, n_actions))
        else:
            layers.append(L.Linear(n_input_channels, n_actions))

        super(FCSoftmaxPolicy, self).__init__(*layers)

    def compute_logits(self, state):
        h = state
        for layer in self[:-1]:
            h = F.relu(layer(h))
        h = self[-1](h)
        return h


def bound_action_by_tanh(action, min_action, max_action):
    assert isinstance(action, chainer.Variable)
    assert min_action is not None
    assert max_action is not None
    xp = cuda.get_array_module(action.data)
    action_scale = (max_action - min_action) / 2
    action_scale = xp.expand_dims(xp.asarray(action_scale), axis=0)
    action_mean = (max_action + min_action) / 2
    action_mean = xp.expand_dims(xp.asarray(action_mean), axis=0)
    return F.tanh(action) * action_scale + action_mean


class FCGaussianPolicy(chainer.ChainList, GaussianPolicy):
    """Gaussian policy that consists of FC layers and rectifiers"""

    def __init__(self, n_input_channels, action_size,
                 n_hidden_layers=0, n_hidden_channels=None,
                 min_action=None, max_action=None, bound_mean=True,
                 clip_action=True, var_type='spherical'):

        self.n_input_channels = n_input_channels
        self.action_size = action_size
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels
        self.min_action = min_action
        self.max_action = max_action
        self.bound_mean = bound_mean
        self.clip_action = clip_action
        var_size = {'spherical': 1, 'diagonal': action_size}[var_type]

        self.hidden_layers = []
        if n_hidden_layers > 0:
            self.hidden_layers.append(
                L.Linear(n_input_channels, n_hidden_channels))
            for i in range(n_hidden_layers - 1):
                self.hidden_layers.append(
                    L.Linear(n_hidden_channels, n_hidden_channels))
            self.mean_layer = L.Linear(n_hidden_channels, action_size)
            self.var_layer = L.Linear(n_hidden_channels, var_size)
        else:
            self.mean_layer = L.Linear(n_input_channels, action_size)
            self.var_layer = L.Linear(n_input_channels, var_size)

        super().__init__(
            self.mean_layer, self.var_layer, *self.hidden_layers)

    def compute_mean_and_var(self, x, test=False):
        h = x
        for layer in self.hidden_layers:
            h = F.relu(layer(h))
        mean = self.mean_layer(h)
        if self.bound_mean:
            mean = bound_action_by_tanh(mean, self.min_action, self.max_action)
        var = F.broadcast_to(F.softplus(self.var_layer(h)), mean.shape)
        return mean, var

    def __call__(self, x, test=False):
        mean, var = self.compute_mean_and_var(x, test=test)
        return policy_output.GaussianPolicyOutput(
            mean, var=var, clip_action=self.clip_action,
            min_action=self.min_action, max_action=self.max_action)


class LinearGaussianPolicyWithDiagonalCovariance(chainer.ChainList, GaussianPolicy):
    """Linear Gaussian policy whose covariance matrix is diagonal."""

    def __init__(self, n_input_channels, action_size):

        self.n_input_channels = n_input_channels
        self.action_size = action_size

        self.mean_layer = L.Linear(n_input_channels, action_size)
        self.var_layer = L.Linear(n_input_channels, action_size)

        super().__init__(self.mean_layer, self.var_layer)

    def compute_mean_and_var(self, x, test=False):
        # mean = self.mean_layer(x)
        mean = F.tanh(self.mean_layer(x)) * 2.0
        var = F.softplus(self.var_layer(x))
        return mean, var


class LinearGaussianPolicyWithSphericalCovariance(chainer.ChainList, GaussianPolicy):
    """Linear Gaussian policy whose covariance matrix is spherical."""

    def __init__(self, n_input_channels, action_size):

        self.n_input_channels = n_input_channels
        self.action_size = action_size

        self.mean_layer = L.Linear(n_input_channels, action_size)
        self.var_layer = L.Linear(n_input_channels, 1)

        super().__init__(self.mean_layer, self.var_layer)

    def compute_mean_and_var(self, x, test=False):
        # mean = self.mean_layer(x)
        mean = F.tanh(self.mean_layer(x)) * 2.0
        var = F.softplus(F.broadcast_to(self.var_layer(x), mean.data.shape))
        return mean, var


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
            a = bound_action_by_tanh(a, self.min_action, self.max_action)

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
            a = bound_action_by_tanh(a, self.min_action, self.max_action)
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


class FCBNDeterministicPolicy(MLPBN, ContinuousDeterministicPolicy):

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

        super().__init__(
            in_size=self.n_input_channels,
            out_size=self.action_size,
            hidden_sizes=[self.n_hidden_channels] * self.n_hidden_layers,
            normalize_input=self.normalize_input)

    def compute_action(self, state, test=False):
        a = super().__call__(state, test=test)

        if self.bound_action:
            a = bound_action_by_tanh(a, self.min_action, self.max_action)

        return a
