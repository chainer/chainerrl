from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

import chainer
from chainer import cuda
import chainer.functions as F

from chainerrl.agents.dqn import DQN
from chainerrl.functions import quantile_loss


class QRDQN(DQN):
    def _compute_loss(self, exp_batch, gamma, errors_out=None):
        xp = self.xp

        y, t = self._compute_y_and_t(exp_batch, gamma)
        _, n_diracs = y.shape
        assert y.shape[0] == t.shape[0]

        # broadcast to (batch, t_n_dirac, y_n_dirac)
        y, t = F.broadcast(y[:,None,:], t[:,:,None])

        # 
        tau_hat = (xp.arange(n_diracs).astype(y.dtype) + 0.5) / n_diracs
        tau_hat = F.broadcast_to(tau_hat, y.shape)

        loss = quantile_loss(y, t, tau_hat)
        loss = F.mean(loss, axis=(1, 2))
        
        if errors_out is not None:
            errors_out[:] = list(cuda.to_cpu(loss.data))

        loss = F.sum(loss)
        return loss

    def _compute_y_and_t(self, exp_batch, gamma):
        batch_size = exp_batch['reward'].shape[0]

        # Compute Q-values for current states
        batch_state = exp_batch['state']

        qout = self.model(batch_state)

        batch_actions = exp_batch['action']
        batch_q = qout.evaluate_actions(batch_actions)

        with chainer.no_backprop_mode():
            batch_q_target = self._compute_target_values(exp_batch, gamma)

        return batch_q, batch_q_target

    def _compute_target_values(self, exp_batch, gamma):

        batch_next_state = exp_batch['next_state']

        target_next_qout = self.target_model(batch_next_state)
        # next_q_max = target_next_qout.max
        next_v = target_next_qout.evaluate_actions(
            target_next_qout.greedy_actions)

        batch_rewards = exp_batch['reward']
        batch_terminal = exp_batch['is_state_terminal']

        shape = next_v.shape
        batch_rewards = F.broadcast_to(batch_rewards[:, None], next_v.shape)
        batch_terminal = F.broadcast_to(batch_terminal[:, None], next_v.shape)

        return batch_rewards + self.gamma * (1.0 - batch_terminal) * next_v
