from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import super
from builtins import range
from future import standard_library
standard_library.install_aliases()
from logging import getLogger
logger = getLogger(__name__)

import chainer
from chainer import functions as F
from chainer import links as L

import policy_output


class Policy(object):
    """Abstract policy class."""

    def __call__(self, state):
        raise NotImplementedError


class SoftmaxPolicy(Policy):
    """Abstract softmax policy class."""

    def compute_logits(self, state):
        """
        Returns:
          ~chainer.Variable: logits of actions
        """
        raise NotImplementedError

    def __call__(self, state):
        return policy_output.SoftmaxPolicyOutput(self.compute_logits(state))


class GaussianPolicy(Policy):
    """Abstract Gaussian policy class.

    You must implement only one of compute_mean_and_ln_var and
    compute_mean_and_var.
    """

    def compute_mean_and_ln_var(self, x):
        """
        Returns:
          tuple of two ~chainer.Variable: mean and log variance
        """
        raise NotImplementedError

    def compute_mean_and_var(self, x):
        """
        Returns:
          tuple of two ~chainer.Variable: mean and variance
        """
        raise NotImplementedError

    def __call__(self, x):
        try:
            mean, ln_var = self.compute_mean_and_ln_var(x)
            return policy_output.GaussianPolicyOutput(mean, ln_var=ln_var)
        except NotImplementedError:
            mean, var = self.compute_mean_and_var(x)
            return policy_output.GaussianPolicyOutput(mean, var=var)


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


class FCGaussianPolicy(chainer.ChainList, GaussianPolicy):
    """Gaussian policy that consists of FC layers and rectifiers"""

    def __init__(self, n_input_channels, action_size,
                 n_hidden_layers=0, n_hidden_channels=None):

        self.n_input_channels = n_input_channels
        self.action_size = action_size
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels

        self.hidden_layers = []
        if n_hidden_layers > 0:
            self.hidden_layers.append(
                L.Linear(n_input_channels, n_hidden_channels))
            for i in range(n_hidden_layers - 1):
                self.hidden_layers.append(
                    L.Linear(n_hidden_channels, n_hidden_channels))
            self.mean_layer = L.Linear(n_hidden_channels, action_size)
            self.ln_var_layer = L.Linear(n_hidden_channels, action_size)
        else:
            self.mean_layer = L.Linear(n_input_channels, action_size)
            self.ln_var_layer = L.Linear(n_input_channels, action_size)

        super().__init__(
            self.mean_layer, self.ln_var_layer, *self.hidden_layers)

    def compute_mean_and_ln_var(self, x):
        h = x
        for layer in self.hidden_layers:
            h = F.relu(layer(h))
        mean = self.mean_layer(x)
        # mean = F.tanh(self.mean_layer(x)) * 2.0
        # ln_var = self.ln_var_layer(x)
        ln_var = F.log(F.softplus(self.ln_var_layer(x)))
        return mean, ln_var


class LinearGaussianPolicyWithDiagonalCovariance(chainer.ChainList, GaussianPolicy):
    """Linear Gaussian policy whose covariance matrix is diagonal."""

    def __init__(self, n_input_channels, action_size):

        self.n_input_channels = n_input_channels
        self.action_size = action_size

        self.mean_layer = L.Linear(n_input_channels, action_size)
        self.var_layer = L.Linear(n_input_channels, 1)

        super().__init__(self.mean_layer, self.var_layer)

    def compute_mean_and_var(self, x):
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

    def compute_mean_and_var(self, x):
        # mean = self.mean_layer(x)
        mean = F.tanh(self.mean_layer(x)) * 2.0
        var = F.softplus(F.broadcast_to(self.var_layer(x), mean.data.shape))
        return mean, var
