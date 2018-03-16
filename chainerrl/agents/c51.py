from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()

import chainer
import chainer.cuda as cuda
import chainer.functions as F

from chainerrl.agents import dqn
from chainerrl.recurrent import state_kept

import numpy as np

class C51(dqn.DQN):
    """Value distribution algorithm.
    """

    def _compute_target_values(self, exp_batch, gamma):
        batch_next_state = exp_batch['next_state']

        # (batch_size, n_actions, n_atoms)
        target_next_qout = self.target_model(batch_next_state)
        next_q_max = target_next_qout.max_distribution.data

        batch_rewards = exp_batch['reward']
        batch_terminal = exp_batch['is_state_terminal']

        batch_size = exp_batch['reward'].shape[0]
        n_atoms = next_q_max.shape[1]
        target_values = self.xp.zeros((batch_size, n_atoms), dtype=self.xp.float32)
        action = exp_batch['action']

        for j in range(n_atoms):
            Tz = self.xp.clip(exp_batch['reward'] + (1.0 - batch_terminal) * gamma * self.z_values[j], self.v_min, self.v_max)
            bj = (Tz - self.v_min) / self.delta_z
            m_l, m_u = self.xp.floor(bj), self.xp.ceil(bj)
            target_values[self.xp.arange(batch_size), m_l.astype(self.xp.int16)] += next_q_max[:, j] * (m_u - bj)
            target_values[self.xp.arange(batch_size), m_u.astype(self.xp.int16)] += next_q_max[:, j] * (bj - m_l)

        return target_values

    def _compute_y_and_t(self, exp_batch, gamma):
        batch_size = exp_batch['reward'].shape[0]

        # Compute Q-values for current states
        batch_state = exp_batch['state']

        # (batch_size, n_actions, n_atoms)
        qout = self.model(batch_state)

        batch_actions = exp_batch['action']
        h = qout.evaluate_actions(
            batch_actions)
        batch_q = F.reshape(h, (batch_size, self.n_atoms))

        with chainer.no_backprop_mode():
            batch_q_target = F.reshape(
                self._compute_target_values(exp_batch, gamma),
                (batch_size, self.n_atoms))

        return batch_q, batch_q_target

    def _compute_loss(self, exp_batch, gamma, errors_out=None):
        """Compute the Q-learning loss for a batch of experiences


        Args:
          experiences (list): see update()'s docstring
          gamma (float): discount factor
        Returns:
          loss
        """

        self.n_atoms = 51
        self.v_min = 0#-10
        self.v_max = 500#10
        self.delta_z = (self.v_max - self.v_min) / float(self.n_atoms - 1)
        self.z_values = self.xp.array([self.v_min + i * self.delta_z for i in range(self.n_atoms)])

        y, t = self._compute_y_and_t(exp_batch, gamma)

        loss = -F.mean(F.sum(t * F.log(y+1e-5), axis=1))

        return loss