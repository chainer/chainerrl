from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

from abc import ABCMeta
from abc import abstractmethod
from abc import abstractproperty

from cached_property import cached_property
import chainer
from chainer import functions as F
from future.utils import with_metaclass
import numpy as np

from chainerrl.functions import mellowmax


def _wrap_by_variable(x):
    if isinstance(x, chainer.Variable):
        return x
    else:
        return chainer.Variable(x)


def _unwrap_variable(x):
    if isinstance(x, chainer.Variable):
        return x.data
    else:
        return x


def sample_discrete_actions(batch_probs):
    """Sample a batch of actions from a batch of action probabilities.

    Args:
        batch_probs (ndarray): batch of action probabilities BxA
    Returns:
        ndarray consisting of sampled action indices
    """
    xp = chainer.cuda.get_array_module(batch_probs)
    return xp.argmax(
        xp.log(batch_probs) + xp.random.gumbel(size=batch_probs.shape),
        axis=1).astype(np.int32, copy=False)


class Distribution(with_metaclass(ABCMeta, object)):
    """Batch of distributions of data."""

    @abstractproperty
    def entropy(self):
        """Entropy of distributions.

        Returns:
            chainer.Variable
        """
        raise NotImplementedError()

    @abstractmethod
    def sample(self):
        """Sample from distributions.

        Returns:
            chainer.Variable
        """
        raise NotImplementedError()

    @abstractmethod
    def prob(self, x):
        """Compute p(x).

        Returns:
            chainer.Variable
        """
        raise NotImplementedError()

    @abstractmethod
    def log_prob(self, x):
        """Compute log p(x).

        Returns:
            chainer.Variable
        """
        raise NotImplementedError()

    @abstractmethod
    def copy(self, x):
        """Copy a distribion unchained from the computation graph.

        Returns:
            Distribution
        """
        raise NotImplementedError()

    @abstractproperty
    def most_probable(self):
        """Most probable data points.

        Returns:
            chainer.Variable
        """
        raise NotImplementedError()

    @abstractproperty
    def kl(self, distrib):
        """Compute KL divergence D_KL(P|Q).

        Args:
            distrib (Distribution): Distribution Q.
        Returns:
            chainer.Variable
        """
        raise NotImplementedError()

    @abstractproperty
    def params(self):
        """Learnable parameters of this distribution.

        Returns:
            tuple of chainer.Variable
        """
        raise NotImplementedError()


class CategoricalDistribution(Distribution):
    """Distribution of categorical data."""

    @cached_property
    def entropy(self):
        with chainer.force_backprop_mode():
            return - F.sum(self.all_prob * self.all_log_prob, axis=1)

    @cached_property
    def most_probable(self):
        return chainer.Variable(
            np.argmax(self.all_prob.data, axis=1).astype(np.int32))

    def sample(self):
        return chainer.Variable(sample_discrete_actions(self.all_prob.data))

    def prob(self, x):
        return F.select_item(self.all_prob, x)

    def log_prob(self, x):
        return F.select_item(self.all_log_prob, x)

    @abstractmethod
    def all_prob(self):
        raise NotImplementedError()

    @abstractmethod
    def all_log_prob(self):
        raise NotImplementedError()

    def kl(self, distrib):
        return F.sum(
            self.all_prob * (self.all_log_prob - distrib.all_log_prob), axis=1)


class SoftmaxDistribution(CategoricalDistribution):
    """Softmax distribution.

    Args:
        logits (ndarray or chainer.Variable): Logits for softmax
            distribution.
    """

    def __init__(self, logits, beta=1.0, min_prob=0.0):
        self.logits = logits
        self.beta = 1.0
        self.min_prob = min_prob
        self.n = logits.shape[1]
        assert self.min_prob * self.n <= 1.0

    @property
    def params(self):
        return (self.logits,)

    @cached_property
    def all_prob(self):
        with chainer.force_backprop_mode():
            if self.min_prob > 0:
                return (F.softmax(self.beta * self.logits)
                        * (1 - self.min_prob * self.n)) + self.min_prob
            else:
                return F.softmax(self.beta * self.logits)

    @cached_property
    def all_log_prob(self):
        with chainer.force_backprop_mode():
            if self.min_prob > 0:
                return F.log(self.all_prob)
            else:
                return F.log_softmax(self.beta * self.logits)

    def copy(self):
        return SoftmaxDistribution(_unwrap_variable(self.logits).copy(),
                                   beta=self.beta)

    def __repr__(self):
        return 'SoftmaxDistribution(beta={}) logits:{} probs:{} entropy:{}'.format(  # NOQA
            self.beta, self.logits.data, self.all_prob.data, self.entropy.data)

    def __getitem__(self, i):
        return SoftmaxDistribution(self.logits[i], beta=self.beta)


class MellowmaxDistribution(CategoricalDistribution):
    """Maximum entropy mellowmax distribution.

    See: http://arxiv.org/abs/1612.05628

    Args:
        values (ndarray or chainer.Variable): Values to apply mellowmax.
    """

    def __init__(self, values, omega=8.):
        self.values = values
        self.omega = omega

    @property
    def params(self):
        return (self.values,)

    @cached_property
    def all_prob(self):
        with chainer.force_backprop_mode():
            return mellowmax.maximum_entropy_mellowmax(self.values)

    @cached_property
    def all_log_prob(self):
        with chainer.force_backprop_mode():
            return F.log(self.all_prob)

    def copy(self):
        return MellowmaxDistribution(_unwrap_variable(self.values).copy(),
                                     omega=self.omega)

    def __repr__(self):
        return 'MellowmaxDistribution(omega={}) values:{} probs:{} entropy:{}'.format(  # NOQA
            self.omega, self.values.data, self.all_prob.data,
            self.entropy.data)

    def __getitem__(self, i):
        return MellowmaxDistribution(self.values[i], omega=self.omega)


def clip_actions(actions, min_action, max_action):
    min_actions = F.broadcast_to(min_action, actions.shape)
    max_actions = F.broadcast_to(max_action, actions.shape)
    return F.maximum(F.minimum(actions, max_actions), min_actions)


class GaussianDistribution(Distribution):
    """Gaussian distribution."""

    def __init__(self, mean, var):
        self.mean = _wrap_by_variable(mean)
        self.var = _wrap_by_variable(var)
        self.ln_var = F.log(var)

    @property
    def params(self):
        return (self.mean, self.var)

    @cached_property
    def most_probable(self):
        return self.mean

    def sample(self):
        return F.gaussian(self.mean, self.ln_var)

    def prob(self, x):
        return F.exp(self.log_prob(x))

    def log_prob(self, x):
        # log N(x|mean,var)
        #   = -0.5log(2pi) - 0.5log(var) - (x - mean)**2 / (2*var)
        log_probs = -0.5 * np.log(2 * np.pi) - \
            0.5 * self.ln_var - \
            ((x - self.mean) ** 2) / (2 * self.var)
        return F.sum(log_probs, axis=1)

    @cached_property
    def entropy(self):
        # Differential entropy of Gaussian is:
        #   0.5 * (log(2 * pi * var) + 1)
        #   = 0.5 * (log(2 * pi) + log var + 1)
        with chainer.force_backprop_mode():
            return 0.5 * self.mean.data.shape[1] * (np.log(2 * np.pi) + 1) + \
                0.5 * F.sum(self.ln_var, axis=1)

    def copy(self):
        return GaussianDistribution(_unwrap_variable(self.mean).copy(),
                                    _unwrap_variable(self.var).copy())

    def kl(self, q):
        p = self
        return 0.5 * F.sum(q.ln_var - p.ln_var +
                           (p.var + (p.mean - q.mean) ** 2) / q.var -
                           1, axis=1)

    def __repr__(self):
        return 'GaussianDistribution mean:{} ln_var:{} entropy:{}'.format(
            self.mean.data, self.ln_var.data, self.entropy.data)

    def __getitem__(self, i):
        return GaussianDistribution(self.mean[i], self.var[i])


class ContinuousDeterministicDistribution(Distribution):
    """Continous deterministic distribution.

    This distribution is supposed to be used in continuous deterministic
    policies.
    """

    def __init__(self, x):
        self.x = _wrap_by_variable(x)

    @cached_property
    def entropy(self):
        raise RuntimeError('Not defined')

    @cached_property
    def most_probable(self):
        return self.x

    def sample(self):
        return self.x

    def prob(self, x):
        raise RuntimeError('Not defined')

    def copy(self):
        return ContinuousDeterministicDistribution(
            _unwrap_variable(self.x).copy())

    def log_prob(self, x):
        raise RuntimeError('Not defined')

    def kl(self, distrib):
        raise RuntimeError('Not defined')

    @property
    def params(self):
        return (self.x,)
