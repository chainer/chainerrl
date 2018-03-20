from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()

import chainer
import chainer.functions as F
import numpy as np

from chainerrl.agents import dqn


class C51(dqn.DQN):
    """Categorical DQN.

    See https://arxiv.org/abs/1707.06887.
    """

    def _compute_target_values(self, exp_batch, gamma):
        """Compute a batch of target return distributions."""

        batch_next_state = exp_batch['next_state']
        xp = self.xp

        target_next_qout = self.target_model(batch_next_state)

        batch_rewards = exp_batch['reward']
        batch_terminal = exp_batch['is_state_terminal']

        batch_size = exp_batch['reward'].shape[0]
        z_values = target_next_qout.z_values
        delta_z = z_values[1] - z_values[0]
        v_min = z_values[0]
        v_max = z_values[-1]
        n_atoms = z_values.size

        # next_q_max: (batch_size, n_atoms)
        next_q_max = target_next_qout.max_as_distribution.data
        assert next_q_max.shape == (batch_size, n_atoms), next_q_max.shape

        # Tz: (batch_size, n_atoms)
        Tz = xp.clip(
            batch_rewards[..., None]
            + (1.0 - batch_terminal[..., None]) * gamma * z_values[None],
            v_min, v_max)
        assert Tz.shape == (batch_size, n_atoms)
        # bj: (batch_size, n_atoms)
        bj = (Tz - v_min) / delta_z
        assert bj.shape == (batch_size, n_atoms)
        # m_l, m_u: (batch_size, n_atoms)
        m_l, m_u = xp.floor(bj), xp.ceil(bj)
        assert m_l.shape == (batch_size, n_atoms)
        assert m_u.shape == (batch_size, n_atoms)

        if xp is chainer.cuda.cupy:
            scatter_add = xp.scatter_add
        else:
            scatter_add = np.add.at
        target_values = xp.zeros((batch_size, n_atoms), dtype=xp.float32)
        offset = xp.arange(
            0, batch_size * n_atoms, n_atoms, dtype=xp.int32)[..., None]
        scatter_add(
            target_values.ravel(),
            (m_l.astype(xp.int32) + offset).ravel(),
            (next_q_max * (m_u - bj)).ravel())
        scatter_add(
            target_values.ravel(),
            (m_u.astype(xp.int32) + offset).ravel(),
            (next_q_max * (bj - m_l)).ravel())
        return target_values

    def _compute_y_and_t(self, exp_batch, gamma):
        """Compute a batch of predicted/target return distributions."""

        batch_size = exp_batch['reward'].shape[0]

        # Compute Q-values for current states
        batch_state = exp_batch['state']

        # (batch_size, n_actions, n_atoms)
        qout = self.model(batch_state)
        n_atoms = qout.z_values.size

        batch_actions = exp_batch['action']
        batch_q = qout.evaluate_actions_as_distribution(batch_actions)
        # batch_q = F.reshape(h, (batch_size, n_atoms))
        assert batch_q.shape == (batch_size, n_atoms)

        with chainer.no_backprop_mode():
            batch_q_target = self._compute_target_values(exp_batch, gamma)
            assert batch_q_target.shape == (batch_size, n_atoms)

        return batch_q, batch_q_target

    def _compute_loss(self, exp_batch, gamma, errors_out=None):
        """Compute a loss of categorical DQN."""
        y, t = self._compute_y_and_t(exp_batch, gamma)
        eltwise_loss = -t * F.log(y)
        if self.batch_accumulator == 'sum':
            loss = F.sum(eltwise_loss)
        else:
            loss = F.mean(F.sum(eltwise_loss, axis=1))
        return loss
