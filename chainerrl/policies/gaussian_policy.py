from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

from abc import abstractmethod
from logging import getLogger
logger = getLogger(__name__)

import chainer
from chainer import functions as F
from chainer import links as L

from chainerrl import distribution
from chainerrl.functions.bound_by_tanh import bound_by_tanh
from chainerrl.policy import Policy


class GaussianPolicy(Policy):
    """Abstract Gaussian policy."""

    @abstractmethod
    def compute_mean_and_var(self, x, test=False):
        """Compute mean and variance.

        Returns:
          tuple of two ~chainer.Variable: mean and variance
        """
        raise NotImplementedError()

    def __call__(self, x, test=False):
        mean, var = self.compute_mean_and_var(x, test=test)
        return distribution.GaussianDistribution(mean=mean, var=var)


class FCGaussianPolicy(chainer.ChainList, GaussianPolicy):
    """Gaussian policy that consists of FC layers and rectifiers"""

    def __init__(self, n_input_channels, action_size,
                 n_hidden_layers=0, n_hidden_channels=None,
                 min_action=None, max_action=None, bound_mean=False,
                 var_type='spherical'):

        self.n_input_channels = n_input_channels
        self.action_size = action_size
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels
        self.min_action = min_action
        self.max_action = max_action
        self.bound_mean = bound_mean
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
            mean = bound_by_tanh(mean, self.min_action, self.max_action)
        var = F.broadcast_to(F.softplus(self.var_layer(h)), mean.shape)
        return mean, var

    def __call__(self, x, test=False):
        mean, var = self.compute_mean_and_var(x, test=test)
        return distribution.GaussianDistribution(mean, var=var)


class LinearGaussianPolicyWithDiagonalCovariance(
        chainer.ChainList, GaussianPolicy):
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


class LinearGaussianPolicyWithSphericalCovariance(
        chainer.ChainList, GaussianPolicy):
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
