import chainer
import chainer.functions as F
from chainer import cuda
import numpy as np

from agents.dqn import DQN


class DPP(DQN):
    """Dynamic Policy Programming."""

    def __init__(self, *args, **kwargs):
        self.eta = kwargs.pop('eta', 1.0)
        super().__init__(*args, **kwargs)

    def _l_operator(self, qout):
        return qout.compute_expectation(self.eta)

    def _compute_target_values(self, experiences, gamma):

        batch_next_state = self._batch_states(
            [elem['next_state'] for elem in experiences])

        target_next_qout = self.target_q_function(batch_next_state, test=True)
        next_q_expect = self._l_operator(target_next_qout)

        batch_rewards = chainer.Variable(self.xp.asarray(
            [elem['reward'] for elem in experiences], dtype=np.float32))

        batch_non_terminal = chainer.Variable(self.xp.asarray(
            [not elem['is_state_terminal'] for elem in experiences],
            dtype=np.float32))

        return batch_rewards + self.gamma * batch_non_terminal * next_q_expect

    def _compute_y_and_t(self, experiences, gamma):

        batch_size = len(experiences)

        batch_state = self._batch_states(
            [elem['state'] for elem in experiences])

        qout = self.q_function(batch_state, test=False)
        xp = cuda.get_array_module(qout.greedy_actions.data)

        batch_actions = chainer.Variable(
            xp.asarray([elem['action'] for elem in experiences]))
        batch_q = F.reshape(qout.evaluate_actions(
            batch_actions), (batch_size, 1))

        # Compute target values
        target_qout = self.target_q_function(batch_state, test=True)

        target_q = F.reshape(target_qout.evaluate_actions(
            batch_actions), (batch_size, 1))

        target_q_expect = F.reshape(
            self._l_operator(target_qout), (batch_size, 1))

        batch_q_target = F.reshape(
            self._compute_target_values(experiences, gamma), (batch_size, 1))

        t = target_q + batch_q_target - target_q_expect
        t.unchain_backward()

        return batch_q, t


class DPPL(DPP):
    """Dynamic Policy Programming."""

    def _l_operator(self, qout):
        return F.logsumexp(self.eta * qout.q_values, axis=1) / self.eta


class DPPGreedy(DPP):
    """Dynamic Policy Programming."""

    def _l_operator(self, qout):
        return qout.max
