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
import numpy as np

from chainerrl import distribution
from chainerrl.functions.bound_by_tanh import bound_by_tanh
from chainerrl.initializers import LeCunNormal
from chainerrl import links
from chainerrl.policy import Policy


class GaussianPolicy(Policy):
    """Abstract Gaussian policy."""

    @abstractmethod
    def compute_mean_and_var(self, x):
        """Compute mean and variance.

        Returns:
          tuple of two ~chainer.Variable: mean and variance
        """
        raise NotImplementedError()

    def __call__(self, x):
        mean, var = self.compute_mean_and_var(x)
        return distribution.GaussianDistribution(mean=mean, var=var)


class FCGaussianPolicy(chainer.ChainList, GaussianPolicy):
    """Gaussian policy that consists of fully-connected layers.

    This model has two output layers: the mean layer and the variance layer.
    The mean of the Gaussian is computed as follows:
        Let y as the output of the mean layer.
        If bound_mean=False:
            mean = y (if bound_mean=False)
        If bound_mean=True:
            mean = min_action + tanh(y) * (max_action - min_action) / 2
    The variance of the Gaussian is computed as follows:
        Let y as the output of the variance layer.
        variance = softplus(y) + min_var

    Args:
        n_input_channels (int): Number of input channels.
        action_size (int): Number of dimensions of the action space.
        n_hidden_layers (int): Number of hidden layers.
        n_hidden_channels (int): Number of hidden channels.
        min_action (ndarray): Minimum action. Used only when bound_mean=True.
        max_action (ndarray): Maximum action. Used only when bound_mean=True.
        var_type (str): Type of parameterization of variance. It must be
            'spherical' or 'diagonal'.
        nonlinearity (callable): Nonlinearity placed between layers.
        mean_wscale (float): Scale of weight initialization of the mean layer.
        var_wscale (float): Scale of weight initialization of the variance
            layer.
        var_bias (float): The initial value of the bias parameter for the
            variance layer.
        min_var (float): Minimum value of the variance.
    """

    def __init__(self, n_input_channels, action_size,
                 n_hidden_layers=0, n_hidden_channels=None,
                 min_action=None, max_action=None, bound_mean=False,
                 var_type='spherical', nonlinearity=F.relu,
                 mean_wscale=1, var_wscale=1, var_bias=0,
                 min_var=0):

        self.n_input_channels = n_input_channels
        self.action_size = action_size
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels
        self.min_action = min_action
        self.max_action = max_action
        self.bound_mean = bound_mean
        self.nonlinearity = nonlinearity
        self.min_var = min_var
        var_size = {'spherical': 1, 'diagonal': action_size}[var_type]

        self.hidden_layers = []
        if n_hidden_layers > 0:
            self.hidden_layers.append(
                L.Linear(n_input_channels, n_hidden_channels))
            for i in range(n_hidden_layers - 1):
                self.hidden_layers.append(
                    L.Linear(n_hidden_channels, n_hidden_channels))
            self.mean_layer = L.Linear(n_hidden_channels, action_size,
                                       initialW=LeCunNormal(mean_wscale))
            self.var_layer = L.Linear(n_hidden_channels, var_size,
                                      initialW=LeCunNormal(var_wscale),
                                      initial_bias=var_bias)
        else:
            self.mean_layer = L.Linear(n_input_channels, action_size,
                                       initialW=LeCunNormal(mean_wscale))
            self.var_layer = L.Linear(n_input_channels, var_size,
                                      initialW=LeCunNormal(var_wscale),
                                      initial_bias=var_bias)

        super().__init__(
            self.mean_layer, self.var_layer, *self.hidden_layers)

    def compute_mean_and_var(self, x):
        h = x
        for layer in self.hidden_layers:
            h = self.nonlinearity(layer(h))
        mean = self.mean_layer(h)
        if self.bound_mean:
            mean = bound_by_tanh(mean, self.min_action, self.max_action)
        var = F.broadcast_to(F.softplus(self.var_layer(h)), mean.shape) + \
            self.min_var
        return mean, var

    def __call__(self, x):
        mean, var = self.compute_mean_and_var(x)
        return distribution.GaussianDistribution(mean, var=var)


class FCGaussianPolicyWithFixedCovariance(links.Sequence, GaussianPolicy):
    """Gaussian policy that consists of FC layers with fixed covariance.

    This model has one output layers: the mean layer.
    The mean of the Gaussian is computed in the same way as FCGaussianPolicy.
    The variance of the Gaussian must be specified as an argument.

    Args:
        n_input_channels (int): Number of input channels.
        action_size (int): Number of dimensions of the action space.
        var (float or ndarray): Variance of the Gaussian distribution.
        n_hidden_layers (int): Number of hidden layers.
        n_hidden_channels (int): Number of hidden channels.
        min_action (ndarray): Minimum action. Used only when bound_mean=True.
        max_action (ndarray): Maximum action. Used only when bound_mean=True.
        var_type (str): Type of parameterization of variance. It must be
            'spherical' or 'diagonal'.
        nonlinearity (callable): Nonlinearity placed between layers.
        mean_wscale (float): Scale of weight initialization of the mean layer.
    """

    def __init__(self, n_input_channels, action_size, var,
                 n_hidden_layers=0, n_hidden_channels=None,
                 min_action=None, max_action=None, bound_mean=False,
                 nonlinearity=F.relu, mean_wscale=1):

        self.n_input_channels = n_input_channels
        self.action_size = action_size
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels
        self.min_action = min_action
        self.max_action = max_action
        self.bound_mean = bound_mean
        self.nonlinearity = nonlinearity
        if np.isscalar(var):
            self.var = np.full(action_size, var, dtype=np.float32)
        else:
            self.var = var
        layers = []
        layers.append(L.Linear(n_input_channels, n_hidden_channels))
        for _ in range(n_hidden_layers - 1):
            layers.append(self.nonlinearity)
            layers.append(L.Linear(n_hidden_channels, n_hidden_channels))
        # The last layer is used to compute the mean
        layers.append(
            L.Linear(n_hidden_channels, action_size,
                     initialW=LeCunNormal(mean_wscale)))

        if self.bound_mean:
            layers.append(lambda x: bound_by_tanh(
                x, self.min_action, self.max_action))

        def get_var_array(shape):
            self.var = self.xp.asarray(self.var)
            return self.xp.broadcast_to(self.var, shape)

        layers.append(lambda x: distribution.GaussianDistribution(
            x, get_var_array(x.shape)))
        super().__init__(*layers)


class LinearGaussianPolicyWithDiagonalCovariance(
        chainer.ChainList, GaussianPolicy):
    """Linear Gaussian policy whose covariance matrix is diagonal."""

    def __init__(self, n_input_channels, action_size):

        self.n_input_channels = n_input_channels
        self.action_size = action_size

        self.mean_layer = L.Linear(n_input_channels, action_size)
        self.var_layer = L.Linear(n_input_channels, action_size)

        super().__init__(self.mean_layer, self.var_layer)

    def compute_mean_and_var(self, x):
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

    def compute_mean_and_var(self, x):
        # mean = self.mean_layer(x)
        mean = F.tanh(self.mean_layer(x)) * 2.0
        var = F.softplus(F.broadcast_to(self.var_layer(x), mean.data.shape))
        return mean, var
