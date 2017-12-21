from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
from future.utils import with_metaclass
standard_library.install_aliases()

from abc import ABCMeta
from abc import abstractmethod
from abc import abstractproperty
import warnings

from cached_property import cached_property
import chainer
from chainer import cuda
from chainer import functions as F
import numpy as np

from chainerrl.misc.chainer_compat import matmul_v3


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
        self.xp = cuda.get_array_module(q_values.data)
        self.q_values = q_values
        self.n_actions = q_values.data.shape[1]
        self.q_values_formatter = q_values_formatter

    @cached_property
    def greedy_actions(self):
        return chainer.Variable(
            self.q_values.data.argmax(axis=1).astype(np.int32))

    @cached_property
    def max(self):
        with chainer.force_backprop_mode():
            return F.select_item(self.q_values, self.greedy_actions)

    def sample_epsilon_greedy_actions(self, epsilon):
        assert self.q_values.data.shape[0] == 1, \
            "This method doesn't support batch computation"
        if np.random.random() < epsilon:
            return chainer.Variable(
                self.xp.asarray([np.random.randint(0, self.n_actions)],
                                dtype=np.int32))
        else:
            return self.greedy_actions

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
            self.greedy_actions.data,
            self.q_values_formatter(self.q_values.data))

    @property
    def params(self):
        return (self.q_values,)


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
        self.xp = cuda.get_array_module(mu.data)
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

        self.batch_size = self.mu.data.shape[0]

    @cached_property
    def greedy_actions(self):
        with chainer.force_backprop_mode():
            a = self.mu
            if self.min_action is not None:
                a = F.maximum(
                    self.xp.broadcast_to(self.min_action, a.data.shape), a)
            if self.max_action is not None:
                a = F.minimum(
                    self.xp.broadcast_to(self.max_action, a.data.shape), a)
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
            matmul_v3(matmul_v3(
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
            self.greedy_actions.data, self.v.data)

    @property
    def params(self):
        return (self.mu, self.mat, self.v)


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
