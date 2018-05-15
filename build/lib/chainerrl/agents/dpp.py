from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

from abc import ABCMeta
from abc import abstractmethod

import chainer
import chainer.functions as F
from future.utils import with_metaclass

from chainerrl.agents.dqn import DQN


class AbstractDPP(with_metaclass(ABCMeta, DQN)):
    """Dynamic Policy Programming.

    See: https://arxiv.org/abs/1004.2027.
    """

    @abstractmethod
    def _l_operator(self, qout):
        raise NotImplementedError()

    def _compute_target_values(self, exp_batch, gamma):

        batch_next_state = exp_batch['next_state']

        target_next_qout = self.target_q_function(batch_next_state)
        next_q_expect = self._l_operator(target_next_qout)

        batch_rewards = exp_batch['reward']
        batch_terminal = exp_batch['is_state_terminal']

        return (batch_rewards +
                self.gamma * (1 - batch_terminal) * next_q_expect)

    def _compute_y_and_t(self, exp_batch, gamma):

        batch_state = exp_batch['state']
        batch_size = len(exp_batch['reward'])

        qout = self.q_function(batch_state)

        batch_actions = exp_batch['action']
        # Q(s_t,a_t)
        batch_q = F.reshape(qout.evaluate_actions(
            batch_actions), (batch_size, 1))

        with chainer.no_backprop_mode():
            # Compute target values
            target_qout = self.target_q_function(batch_state)

            # Q'(s_t,a_t)
            target_q = F.reshape(target_qout.evaluate_actions(
                batch_actions), (batch_size, 1))

            # LQ'(s_t,a)
            target_q_expect = F.reshape(
                self._l_operator(target_qout), (batch_size, 1))

            # r + g * LQ'(s_{t+1},a)
            batch_q_target = F.reshape(
                self._compute_target_values(exp_batch, gamma), (batch_size, 1))

            # Q'(s_t,a_t) + r + g * LQ'(s_{t+1},a) - LQ'(s_t,a)
            t = target_q + batch_q_target - target_q_expect

        return batch_q, t


class DPP(AbstractDPP):
    """Dynamic Policy Programming with softmax operator.

    Args:
      eta (float): Positive constant.

    For other arguments, see DQN.
    """

    def __init__(self, *args, **kwargs):
        self.eta = kwargs.pop('eta', 1.0)
        super().__init__(*args, **kwargs)

    def _l_operator(self, qout):
        return qout.compute_expectation(self.eta)


class DPPL(AbstractDPP):
    """Dynamic Policy Programming with L operator.

    Args:
      eta (float): Positive constant.

    For other arguments, see DQN.
    """

    def __init__(self, *args, **kwargs):
        self.eta = kwargs.pop('eta', 1.0)
        super().__init__(*args, **kwargs)

    def _l_operator(self, qout):
        return F.logsumexp(self.eta * qout.q_values, axis=1) / self.eta


class DPPGreedy(AbstractDPP):
    """Dynamic Policy Programming with max operator.

    This algorithm corresponds to DPP with eta = infinity.
    """

    def _l_operator(self, qout):
        return qout.max
