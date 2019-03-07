from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
from future.utils import with_metaclass
standard_library.install_aliases()  # NOQA

from abc import ABCMeta
from abc import abstractmethod
from abc import abstractproperty
import warnings

from cached_property import cached_property
import chainer
from chainer import cuda
from chainer import functions as F
import numpy as np


class ActionValue(with_metaclass(ABCMeta, object)):
    """Struct that holds state-fixed Q-functions and its subproducts.

    Every operation it supports is done in a batch manner.
    """

    @abstractproperty
    def greedy_actions(self):
        """Get argmax_a Q(s,a)."""
        raise NotImplementedError()

    @abstractproperty
    def max(self):
        """Evaluate max Q(s,a)."""
        raise NotImplementedError()

    @abstractmethod
    def evaluate_actions(self, actions):
        """Evaluate Q(s,a) with a = given actions."""
        raise NotImplementedError()

    @abstractproperty
    def params(self):
        """Learnable parameters of this action value.

        Returns:
            tuple of chainer.Variable
        """
        raise NotImplementedError()


class DiscreteActionValue(ActionValue):
    """Q-function output for discrete action space.

    Args:
        q_values (ndarray or chainer.Variable):
            Array of Q values whose shape is (batchsize, n_actions)
    """

    def __init__(self, q_values, q_values_formatter=lambda x: x):
        assert isinstance(q_values, chainer.Variable)
        self.xp = cuda.get_array_module(q_values.array)
        self.q_values = q_values
        self.n_actions = q_values.array.shape[1]
        self.q_values_formatter = q_values_formatter

    @cached_property
    def greedy_actions(self):
        return chainer.Variable(
            self.q_values.array.argmax(axis=1).astype(np.int32))

    @cached_property
    def max(self):
        with chainer.force_backprop_mode():
            return F.select_item(self.q_values, self.greedy_actions)

    def evaluate_actions(self, actions):
        return F.select_item(self.q_values, actions)

    def compute_advantage(self, actions):
        return self.evaluate_actions(actions) - self.max

    def compute_double_advantage(self, actions, argmax_actions):
        return (self.evaluate_actions(actions) -
                self.evaluate_actions(argmax_actions))

    def compute_expectation(self, beta):
        return F.sum(F.softmax(beta * self.q_values) * self.q_values, axis=1)

    def __repr__(self):
        return 'DiscreteActionValue greedy_actions:{} q_values:{}'.format(
            self.greedy_actions.array,
            self.q_values_formatter(self.q_values.array))

    @property
    def params(self):
        return (self.q_values,)

    def __getitem__(self, i):
        return DiscreteActionValue(
            self.q_values[i], q_values_formatter=self.q_values_formatter)


class DistributionalDiscreteActionValue(ActionValue):
    """distributional Q-function output for discrete action space.

    Args:
        q_dist (chainer.Variable): Probabilities of atoms. Its shape must be
            (batchsize, n_actions, n_atoms).
        z_values (ndarray): Values represented by atoms.
            Its shape must be (n_atoms,).
    """

    def __init__(self, q_dist, z_values, q_values_formatter=lambda x: x):
        assert isinstance(q_dist, chainer.Variable)
        assert not isinstance(z_values, chainer.Variable)
        assert q_dist.ndim == 3
        assert z_values.ndim == 1
        assert q_dist.shape[2] == z_values.shape[0]

        self.xp = cuda.get_array_module(q_dist.array)
        self.z_values = z_values
        self.q_values = F.sum(F.scale(q_dist, self.z_values, axis=2), axis=2)
        self.q_dist = q_dist
        self.n_actions = q_dist.array.shape[1]
        self.q_values_formatter = q_values_formatter

    @cached_property
    def greedy_actions(self):
        return chainer.Variable(
            self.q_values.array.argmax(axis=1).astype(np.int32))

    @cached_property
    def max(self):
        with chainer.force_backprop_mode():
            return F.select_item(self.q_values, self.greedy_actions)

    @cached_property
    def max_as_distribution(self):
        """Return the return distributions of the greedy actions.

        Returns:
            chainer.Variable: Return distributions. Its shape will be
                (batch_size, n_atoms).
        """
        with chainer.force_backprop_mode():
            return self.q_dist[self.xp.arange(self.q_values.shape[0]),
                               self.greedy_actions.array]

    def evaluate_actions(self, actions):
        return F.select_item(self.q_values, actions)

    def evaluate_actions_as_distribution(self, actions):
        """Return the return distributions of given actions.

        Args:
            actions (chainer.Variable or ndarray): Array of action indices.
                Its shape must be (batch_size,).

        Returns:
            chainer.Variable: Return distributions. Its shape will be
                (batch_size, n_atoms).
        """
        return self.q_dist[self.xp.arange(self.q_values.shape[0]), actions]

    def compute_advantage(self, actions):
        return self.evaluate_actions(actions) - self.max

    def compute_double_advantage(self, actions, argmax_actions):
        return (self.evaluate_actions(actions) -
                self.evaluate_actions(argmax_actions))

    def compute_expectation(self, beta):
        return F.sum(F.softmax(beta * self.q_values) * self.q_values, axis=1)

    def __repr__(self):
        return 'DistributionalDiscreteActionValue greedy_actions:{} q_values:{}'.format(  # NOQA
            self.greedy_actions.array,
            self.q_values_formatter(self.q_values.array))

    @property
    def params(self):
        return (self.q_dist,)

    def __getitem__(self, i):
        return DistributionalDiscreteActionValue(
            self.q_dist[i],
            self.z_values,
            q_values_formatter=self.q_values_formatter,
        )


class QuantileDiscreteActionValue(DiscreteActionValue):
    """Quantile action value for discrete actions.

    Args:
        quantiles (chainer.Variable): (batch_size, n_taus, n_actions)
        q_values_formatter (callable):
    """

    def __init__(self, quantiles, q_values_formatter=lambda x: x):
        assert quantiles.ndim == 3
        self.quantiles = quantiles
        self.xp = cuda.get_array_module(quantiles.array)
        self.n_actions = quantiles.shape[2]
        self.q_values_formatter = q_values_formatter

    @cached_property
    def q_values(self):
        with chainer.force_backprop_mode():
            return F.mean(self.quantiles, axis=1)

    def evaluate_actions_as_quantiles(self, actions):
        """Return the return quantiles of given actions.

        Args:
            actions (chainer.Variable or ndarray): Array of action indices.
                Its shape must be (batch_size,).

        Returns:
            chainer.Variable: Return quantiles. Its shape will be
                (batch_size, n_taus).
        """
        if isinstance(actions, chainer.Variable):
            actions = actions.array
        return self.quantiles[
            self.xp.arange(self.quantiles.shape[0]), :, actions]

    def __repr__(self):
        return 'QuantileDiscreteActionValue greedy_actions:{} q_values:{}'.format(  # NOQA
            self.greedy_actions.array,
            self.q_values_formatter(self.q_values.array))

    @property
    def params(self):
        return (self.quantiles,)

    def __getitem__(self, i):
        return QuantileDiscreteActionValue(
            quantiles=self.quantiles[i],
            q_values_formatter=self.q_values_formatter,
        )


class QuadraticActionValue(ActionValue):
    """Q-function output for continuous action space.

    See: http://arxiv.org/abs/1603.00748

    Define a Q(s,a) with A(s,a) in a quadratic form.

    Q(s,a) = V(s,a) + A(s,a)
    A(s,a) = -1/2 (u - mu(s))^T P(s) (u - mu(s))

    Args:
        mu (chainer.Variable): mu(s), actions that maximize A(s,a)
        mat (chainer.Variable): P(s), coefficient matrices of A(s,a).
          It must be positive definite.
        v (chainer.Variable): V(s), values of s
        min_action (ndarray): mininum action, not batched
        max_action (ndarray): maximum action, not batched
    """

    def __init__(self, mu, mat, v, min_action=None, max_action=None):
        self.xp = cuda.get_array_module(mu.array)
        self.mu = mu
        self.mat = mat
        self.v = v
        if min_action is None:
            self.min_action = None
        else:
            self.min_action = self.xp.asarray(min_action, dtype=np.float32)
        if max_action is None:
            self.max_action = None
        else:
            self.max_action = self.xp.asarray(max_action, dtype=np.float32)

        self.batch_size = self.mu.array.shape[0]

    @cached_property
    def greedy_actions(self):
        with chainer.force_backprop_mode():
            a = self.mu
            if self.min_action is not None:
                a = F.maximum(
                    self.xp.broadcast_to(self.min_action, a.array.shape), a)
            if self.max_action is not None:
                a = F.minimum(
                    self.xp.broadcast_to(self.max_action, a.array.shape), a)
            return a

    @cached_property
    def max(self):
        with chainer.force_backprop_mode():
            if self.min_action is None and self.max_action is None:
                return F.reshape(self.v, (self.batch_size,))
            else:
                return self.evaluate_actions(self.greedy_actions)

    def evaluate_actions(self, actions):
        u_minus_mu = actions - self.mu
        a = - 0.5 * \
            F.matmul(F.matmul(
                u_minus_mu[:, None, :], self.mat),
                u_minus_mu[:, :, None])[:, 0, 0]
        return a + F.reshape(self.v, (self.batch_size,))

    def compute_advantage(self, actions):
        return self.evaluate_actions(actions) - self.max

    def compute_double_advantage(self, actions, argmax_actions):
        return (self.evaluate_actions(actions) -
                self.evaluate_actions(argmax_actions))

    def __repr__(self):
        return 'QuadraticActionValue greedy_actions:{} v:{}'.format(
            self.greedy_actions.array, self.v.array)

    @property
    def params(self):
        return (self.mu, self.mat, self.v)

    def __getitem__(self, i):
        return QuadraticActionValue(
            self.mu[i],
            self.mat[i],
            self.v[i],
            min_action=self.min_action,
            max_action=self.max_action,
        )


class SingleActionValue(ActionValue):
    """ActionValue that can evaluate only a single action."""

    def __init__(self, evaluator, maximizer=None):
        self.evaluator = evaluator
        self.maximizer = maximizer

    @cached_property
    def greedy_actions(self):
        with chainer.force_backprop_mode():
            return self.maximizer()

    @cached_property
    def max(self):
        with chainer.force_backprop_mode():
            return self.evaluator(self.greedy_actions)

    def evaluate_actions(self, actions):
        return self.evaluator(actions)

    def compute_advantage(self, actions):
        return self.evaluator(actions) - self.max

    def compute_double_advantage(self, actions, argmax_actions):
        return (self.evaluate_actions(actions) -
                self.evaluate_actions(argmax_actions))

    def __repr__(self):
        return 'SingleActionValue'

    @property
    def params(self):
        warnings.warn(
            'SingleActionValue has no learnable parameters until it'
            ' is evaluated on some action. If you want to draw a computation'
            ' graph that outputs SingleActionValue, use the variable returned'
            ' by its method such as evaluate_actions instead.')
        return ()

    def __getitem__(self, i):
        raise NotImplementedError
