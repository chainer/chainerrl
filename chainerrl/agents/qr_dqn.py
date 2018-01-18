from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

import functools

import chainer
from chainer import cuda
import chainer.functions as F

import chainerrl
from chainerrl.agents.dqn import DQN
from chainerrl.functions import quantile_huber_loss_Dabney


# Definitions here are for discrete actions
class _SingleModelStateQuantileQFunction(
        chainer.Chain, chainerrl.q_function.StateQFunction):

    def __init__(self, model):
        super().__init__()
        with self.init_scope():
            self.model = model

    def __call__(self, x):
        h = self.model(x)
        return _ActionQuantile(h)


class _ActionQuantile(chainerrl.action_value.DiscreteActionValue):

    def __init__(self, q_quantiles, q_values_formatter=lambda x: x):
        assert isinstance(q_quantiles, chainer.Variable)
        assert q_quantiles.ndim == 3
        self.xp = cuda.get_array_module(q_quantiles.data)
        self.q_quantiles = q_quantiles
        self.n_actions = q_quantiles.data.shape[1]
        self.n_diracs = q_quantiles.data.shape[2]
        self.q_values_formatter = q_values_formatter

    @property
    def q_values(self):
        return F.mean(self.q_quantiles, axis=2)

    def evaluate_actions(self, actions):
        if isinstance(actions, chainer.Variable):
            actions = actions.data

        return self.q_quantiles[
            self.xp.arange(actions.size),
            actions
        ]


class FCQuantileQFunction(_SingleModelStateQuantileQFunction):
    def __init__(
            self,
            obs_size, n_actions, n_diracs,
            n_hidden_channels,
            n_hidden_layers,
    ):
        super().__init__(
            model=chainerrl.links.Sequence(
                chainerrl.links.MLP(
                    obs_size, n_actions * n_diracs,
                    hidden_sizes=[n_hidden_channels] * n_hidden_layers,
                ),
                functools.partial(
                    F.reshape,
                    shape=(-1, n_actions, n_diracs)
                )
            )
        )


class QRDQN(DQN):
    """Quantile Regression DQN

    Args:
        q_function (Chain): Quantile Q-function
            The output of q_function should be ActionValue and
            its evaluate_actions should return an array of quantiles
            of shape (minibatch_size, n) where the quantiles are represented
            as sums of n dirac distributions.
        args of DQN

    See: https://arxiv.org/abs/1710.10044
    """

    def _compute_loss(self, exp_batch, gamma, errors_out=None):
        xp = self.xp

        y, t = self._compute_y_and_t(exp_batch, gamma)
        _, n_diracs = y.shape
        assert y.shape[0] == t.shape[0]

        # broadcast to (batch, t_n_dirac, y_n_dirac)
        y, t = F.broadcast(y[:, None, :], t[:, :, None])

        tau_hat = (xp.arange(n_diracs).astype(y.dtype) + 0.5) / n_diracs
        tau_hat = F.broadcast_to(tau_hat, y.shape)

        # loss = quantile_loss(y, t, tau_hat)
        loss = quantile_huber_loss_Dabney(y, t, tau_hat)
        loss = F.mean(loss, axis=(1, 2))

        if errors_out is not None:
            errors_out[:] = list(cuda.to_cpu(loss.data))

        loss = F.sum(loss)
        return loss

    def _compute_y_and_t(self, exp_batch, gamma):
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
        batch_rewards = F.broadcast_to(batch_rewards[:, None], shape)
        batch_terminal = F.broadcast_to(batch_terminal[:, None], shape)

        return batch_rewards + self.gamma * (1.0 - batch_terminal) * next_v
