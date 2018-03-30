from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()

import chainer
import chainer.functions as F

from chainerrl.agents import dqn


def _apply_categorical_projection(y, y_probs, z):
    """Apply categorical projection.

    See (7) in https://arxiv.org/abs/1707.06887.

    Args:
        y (ndarray): Values of atoms before projection. Its shape must be
            (batch_size, n_atoms).
        y_probs (ndarray): Probabilities of atoms whose values are y.
            Its shape must be (batch_size, n_atoms).
        z (ndarray): Values of atoms before projection after projection. Its
            shape must be (n_atoms,). It is assumed that the values are sorted
            in ascending order and evenly spaced.

    Returns:
        ndarray: Probabilities of atoms whose values are z.
    """
    batch_size, n_atoms = y.shape
    assert z.shape == (n_atoms,)
    assert y_probs.shape == (batch_size, n_atoms)
    delta_z = z[1] - z[0]
    v_min = z[0]
    v_max = z[-1]
    xp = chainer.cuda.get_array_module(z)
    y = xp.clip(y, v_min, v_max)
    # Broadcast to (batch_size, n_atoms, n_atoms) to consider all the
    # combinations of z and y. The second and third axes correspond to z and y,
    # respectively.
    y = y.reshape((batch_size, 1, n_atoms))
    y_probs = y_probs.reshape((batch_size, 1, n_atoms))
    z = z.reshape((1, n_atoms, 1))
    return (xp.clip(1 - abs(y - z) / delta_z, 0, 1) * y_probs).sum(axis=2)


class C51(dqn.DQN):
    """Categorical DQN.

    See https://arxiv.org/abs/1707.06887.
    """

    def _compute_target_values(self, exp_batch, gamma):
        """Compute a batch of target return distributions."""

        batch_next_state = exp_batch['next_state']
        target_next_qout = self.target_model(batch_next_state)

        batch_rewards = exp_batch['reward']
        batch_terminal = exp_batch['is_state_terminal']

        batch_size = exp_batch['reward'].shape[0]
        z_values = target_next_qout.z_values
        n_atoms = z_values.size

        # next_q_max: (batch_size, n_atoms)
        next_q_max = target_next_qout.max_as_distribution.data
        assert next_q_max.shape == (batch_size, n_atoms), next_q_max.shape

        # Tz: (batch_size, n_atoms)
        Tz = (batch_rewards[..., None]
              + (1.0 - batch_terminal[..., None]) * gamma * z_values[None])
        return _apply_categorical_projection(Tz, next_q_max, z_values)

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
        # minimize the cross entropy
        eltwise_loss = -t * F.log(F.clip(y, 1e-10, 1.))
        if self.batch_accumulator == 'sum':
            loss = F.sum(eltwise_loss)
        else:
            loss = F.mean(F.sum(eltwise_loss, axis=1))
        return loss
