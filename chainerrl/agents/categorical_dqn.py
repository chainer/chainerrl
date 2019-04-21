from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()  # NOQA

import chainer
from chainer import cuda
import chainer.functions as F
import numpy as np

from chainerrl.agents import dqn


def _apply_categorical_projection(y, y_probs, z):
    """Apply categorical projection.

    See Algorithm 1 in https://arxiv.org/abs/1707.06887.

    Args:
        y (ndarray): Values of atoms before projection. Its shape must be
            (batch_size, n_atoms).
        y_probs (ndarray): Probabilities of atoms whose values are y.
            Its shape must be (batch_size, n_atoms).
        z (ndarray): Values of atoms after projection. Its shape must be
            (n_atoms,). It is assumed that the values are sorted in ascending
            order and evenly spaced.

    Returns:
        ndarray: Probabilities of atoms whose values are z.
    """
    batch_size, n_atoms = y.shape
    assert z.shape == (n_atoms,)
    assert y_probs.shape == (batch_size, n_atoms)
    delta_z = z[1] - z[0]
    v_min = z[0]
    v_max = z[-1]
    xp = cuda.get_array_module(z)
    y = xp.clip(y, v_min, v_max)

    # bj: (batch_size, n_atoms)
    bj = (y - v_min) / delta_z
    assert bj.shape == (batch_size, n_atoms)
    # Avoid the error caused by inexact delta_z
    bj = xp.clip(bj, 0, n_atoms - 1)

    # l, u: (batch_size, n_atoms)
    l, u = xp.floor(bj), xp.ceil(bj)
    assert l.shape == (batch_size, n_atoms)
    assert u.shape == (batch_size, n_atoms)

    if cuda.available and xp is cuda.cupy:
        scatter_add = xp.scatter_add
    else:
        scatter_add = np.add.at

    z_probs = xp.zeros((batch_size, n_atoms), dtype=xp.float32)
    offset = xp.arange(
        0, batch_size * n_atoms, n_atoms, dtype=xp.int32)[..., None]
    # Accumulate m_l
    # Note that u - bj in the original paper is replaced with 1 - (bj - l) to
    # deal with the case when bj is an integer, i.e., l = u = bj
    scatter_add(
        z_probs.ravel(),
        (l.astype(xp.int32) + offset).ravel(),
        (y_probs * (1 - (bj - l))).ravel())
    # Accumulate m_u
    scatter_add(
        z_probs.ravel(),
        (u.astype(xp.int32) + offset).ravel(),
        (y_probs * (bj - l)).ravel())
    return z_probs


def compute_value_loss(eltwise_loss, batch_accumulator='mean'):
    """Compute a loss for value prediction problem.

    Args:
        eltwise_loss (Variable): Element-wise loss per example per atom
        batch_accumulator (str): 'mean' or 'sum'. 'mean' will use the mean of
            the loss values in a batch. 'sum' will use the sum.
    Returns:
        (Variable) scalar loss
    """
    assert batch_accumulator in ('mean', 'sum')

    if batch_accumulator == 'sum':
        loss = F.sum(eltwise_loss)
    else:
        loss = F.mean(F.sum(eltwise_loss, axis=1))
    return loss


def compute_weighted_value_loss(eltwise_loss, batch_size, weights,
                                batch_accumulator='mean'):
    """Compute a loss for value prediction problem.

    Args:
        eltwise_loss (Variable): Element-wise loss per example per atom
        weights (ndarray): Weights for y, t.
        batch_accumulator (str): 'mean' will divide loss by batchsize
    Returns:
        (Variable) scalar loss
    """
    assert batch_accumulator in ('mean', 'sum')

    # eltwise_loss is (batchsize, n_atoms) array of losses
    # weights is an array of shape (batch_size)
    # sum loss across atoms and then apply weight per example in batch
    loss_sum = F.matmul(F.sum(eltwise_loss, axis=1), weights)
    if batch_accumulator == 'mean':
        loss = loss_sum / batch_size
    elif batch_accumulator == 'sum':
        loss = loss_sum
    return loss


class CategoricalDQN(dqn.DQN):
    """Categorical DQN.

    See https://arxiv.org/abs/1707.06887.

    Arguments are the same as those of DQN except q_function must return
    DistributionalDiscreteActionValue and clip_delta is ignored.
    """

    def _compute_target_values(self, exp_batch):
        """Compute a batch of target return distributions."""

        batch_next_state = exp_batch['next_state']
        target_next_qout = self.target_model(batch_next_state)

        batch_rewards = exp_batch['reward']
        batch_terminal = exp_batch['is_state_terminal']

        batch_size = exp_batch['reward'].shape[0]
        z_values = target_next_qout.z_values
        n_atoms = z_values.size

        # next_q_max: (batch_size, n_atoms)
        next_q_max = target_next_qout.max_as_distribution.array
        assert next_q_max.shape == (batch_size, n_atoms), next_q_max.shape

        # Tz: (batch_size, n_atoms)
        Tz = (batch_rewards[..., None]
              + (1.0 - batch_terminal[..., None])
              * self.xp.expand_dims(exp_batch['discount'], 1)
              * z_values[None])
        return _apply_categorical_projection(Tz, next_q_max, z_values)

    def _compute_y_and_t(self, exp_batch):
        """Compute a batch of predicted/target return distributions."""

        batch_size = exp_batch['reward'].shape[0]

        # Compute Q-values for current states
        batch_state = exp_batch['state']

        # (batch_size, n_actions, n_atoms)
        qout = self.model(batch_state)
        n_atoms = qout.z_values.size

        batch_actions = exp_batch['action']
        batch_q = qout.evaluate_actions_as_distribution(batch_actions)
        assert batch_q.shape == (batch_size, n_atoms)

        with chainer.no_backprop_mode():
            batch_q_target = self._compute_target_values(exp_batch)
            assert batch_q_target.shape == (batch_size, n_atoms)

        return batch_q, batch_q_target

    def _compute_loss(self, exp_batch, errors_out=None):
        """Compute a loss of categorical DQN."""
        y, t = self._compute_y_and_t(exp_batch)
        # Minimize the cross entropy
        # y is clipped to avoid log(0)
        eltwise_loss = -t * F.log(F.clip(y, 1e-10, 1.))

        if errors_out is not None:
            del errors_out[:]
            # The loss per example is the sum of the atom-wise loss
            # Prioritization by KL-divergence
            delta = F.sum(eltwise_loss, axis=1)
            delta = cuda.to_cpu(delta.array)
            for e in delta:
                errors_out.append(e)

        if 'weights' in exp_batch:
            return compute_weighted_value_loss(
                eltwise_loss, y.shape[0], exp_batch['weights'],
                batch_accumulator=self.batch_accumulator)
        else:
            return compute_value_loss(
                eltwise_loss, batch_accumulator=self.batch_accumulator)
