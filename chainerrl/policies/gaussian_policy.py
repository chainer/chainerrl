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
import numpy as np

from chainerrl import distribution
from chainerrl.functions.bound_by_tanh import bound_by_tanh
from chainerrl import links
from chainerrl.policy import Policy


logger = getLogger(__name__)


class FCGaussianPolicy(chainer.ChainList, Policy):
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


class FCGaussianPolicyWithStateIndependentCovariance(
        chainer.Chain, Policy):
    """Gaussian policy that consists of FC layers with parametrized covariance.

    This model has one output layers: the mean layer.
    The mean of the Gaussian is computed in the same way as FCGaussianPolicy.

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
        var_func (callable): Callable that computes the variance from the var
            parameter. It should always return positive values.
        var_param_init (float): Initial value the var parameter.
    """

    def __init__(self, n_input_channels, action_size,
                 n_hidden_layers=0, n_hidden_channels=None,
                 min_action=None, max_action=None, bound_mean=False,
                 var_type='spherical',
                 nonlinearity=F.relu,
                 mean_wscale=1,
                 var_func=F.softplus,
                 var_param_init=0,
                 ):

        self.n_input_channels = n_input_channels
        self.action_size = action_size
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels
        self.min_action = min_action
        self.max_action = max_action
        self.bound_mean = bound_mean
        self.nonlinearity = nonlinearity
        self.var_func = var_func
        var_size = {'spherical': 1, 'diagonal': action_size}[var_type]

        layers = []
        layers.append(L.Linear(n_input_channels, n_hidden_channels))
        for _ in range(n_hidden_layers - 1):
            layers.append(self.nonlinearity)
            layers.append(L.Linear(n_hidden_channels, n_hidden_channels))
        layers.append(self.nonlinearity)
        # The last layer is used to compute the mean
        layers.append(
            L.Linear(n_hidden_channels, action_size,
                     initialW=LeCunNormal(mean_wscale)))

        if self.bound_mean:
            layers.append(lambda x: bound_by_tanh(
                x, self.min_action, self.max_action))

        super().__init__()
        with self.init_scope():
            self.hidden_layers = links.Sequence(*layers)
            self.var_param = chainer.Parameter(
                initializer=var_param_init, shape=(var_size,))

    def __call__(self, x):
        mean = self.hidden_layers(x)
        var = F.broadcast_to(self.var_func(self.var_param), mean.shape)
        return distribution.GaussianDistribution(mean, var)


class FCGaussianPolicyWithFixedCovariance(links.Sequence, Policy):
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
        if n_hidden_layers > 0:
            # Input to hidden
            layers.append(L.Linear(n_input_channels, n_hidden_channels))
            layers.append(self.nonlinearity)
            for _ in range(n_hidden_layers - 1):
                # Hidden to hidden
                layers.append(L.Linear(n_hidden_channels, n_hidden_channels))
                layers.append(self.nonlinearity)
            # The last layer is used to compute the mean
            layers.append(
                L.Linear(n_hidden_channels, action_size,
                         initialW=LeCunNormal(mean_wscale)))
        else:
            # There's only one layer for computing the mean
            layers.append(
                L.Linear(n_input_channels, action_size,
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


class GaussianHeadWithStateIndependentCovariance(chainer.Chain):
    """Gaussian head with state-independent learned covariance.

    This link is intended to be attached to a neural network that outputs
    the mean of a Gaussian policy. The only learnable parameter this link has
    determines the variance in a state-independent way.

    State-independent parameterization of the variance of a Gaussian policy
    is often used with PPO and TRPO, e.g., in https://arxiv.org/abs/1709.06560.

    Args:
        action_size (int): Number of dimensions of the action space.
        var_type (str): Type of parameterization of variance. It must be
            'spherical' or 'diagonal'.
        var_func (callable): Callable that computes the variance from the var
            parameter. It should always return positive values.
        var_param_init (float): Initial value the var parameter.
    """

    def __init__(
            self,
            action_size,
            var_type='spherical',
            var_func=F.softplus,
            var_param_init=0,
    ):

        self.var_func = var_func
        var_size = {'spherical': 1, 'diagonal': action_size}[var_type]

        super().__init__()
        with self.init_scope():
            self.var_param = chainer.Parameter(
                initializer=var_param_init, shape=(var_size,))

    def __call__(self, mean):
        """Return a Gaussian with given mean.

        Args:
            mean (chainer.Variable or ndarray): Mean of Gaussian.

        Returns:
            chainerrl.distribution.Distribution: Gaussian whose mean is the
                mean argument and whose variance is computed from the parameter
                of this link.
        """
        var = F.broadcast_to(self.var_func(self.var_param), mean.shape)
        return distribution.GaussianDistribution(mean, var)
