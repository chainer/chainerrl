from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
import numpy as np

from chainerrl.action_value import QuantileDiscreteActionValue
from chainerrl.agents import dqn


class CosineBasisLinearReLU(chainer.Chain):

    def __init__(self, n_bases, out_size):
        super().__init__()
        with self.init_scope():
            self.linear = L.Linear(n_bases, out_size)
        self.n_bases = n_bases
        self.out_size = out_size

    def __call__(self, taus):
        """Evaluate.

        Args:
            taus (chainer.Variable or ndarray): Quantile thresholds
                (batch_size, n_taus).
        Returns:
            chainer.Variable: (batch_size, n_taus, hidden_size).
        """
        batch_size, n_taus = taus.shape
        taus = F.reshape(taus, (-1, 1))
        taus = F.broadcast_to(taus, (batch_size * n_taus, self.n_bases))
        xp = self.xp
        coef = xp.arange(self.n_bases, dtype=np.float32) * np.pi
        coef = xp.broadcast_to(coef, (batch_size * n_taus, self.n_bases))
        out = F.relu(self.linear(F.cos(coef * taus)))
        return F.reshape(out, (batch_size, n_taus, self.out_size))


class ImplicitQuantileQFunction(chainer.Chain):
    """Implicit quantile network-based Q-function.

    Args:
        psi (chainer.Link): Callable link
            (batch_size, obs_size) -> (batch_size, hidden_size).
        phi (chainer.Link): Callable link
            (batch_size, n_taus) -> (batch_size, n_taus, hidden_size).
        f (chainer.Link): Callable link
            (batch_size * n_taus, hidden_size)
            -> (batch_size * n_taus, n_actions).

    Returns:
        ImplicitQuantileDiscreteActionValue: Action values.
    """

    def __init__(self, psi, phi, f):
        super().__init__()
        with self.init_scope():
            self.psi = psi
            self.phi = phi
            self.f = f

    def __call__(self, x):
        """Evaluate given observations.

        Args:
            x (ndarray): Batch of observations.
        Returns:
            callable: (batch_size, taus) -> (batch_size, taus, n_actions)
        """
        batch_size = x.shape[0]
        psi_x = self.psi(x)
        hidden_size = psi_x.shape[1]

        def evaluate_with_quantile_thresholds(taus):
            assert taus.ndim == 2
            assert taus.shape[0] == batch_size
            n_taus = taus.shape[1]
            phi_taus = self.phi(taus)
            psi_x_b = F.broadcast_to(
                F.expand_dims(psi_x, axis=1), phi_taus.shape)
            h = psi_x_b * phi_taus
            h = F.reshape(h, (-1, hidden_size))
            h = self.f(h)
            n_actions = h.shape[-1]
            h = F.reshape(h, (batch_size, n_taus, n_actions))
            return QuantileDiscreteActionValue(h)

        return evaluate_with_quantile_thresholds


def _unwrap_variable(x):
    if isinstance(x, chainer.Variable):
        return x.data
    else:
        return x


def compute_eltwise_huber_quantile_loss(y, t, taus):
    """Compute elementwise Huber losses for quantile regression.

    Args:
        y (chainer.Variable): (batch_size, N, N_prime)
        t (chainer.Variable or ndarray): (batch_size, N, N_prime)
        taus (ndarray): (batch_size, N)
    """
    with chainer.no_backprop_mode():
        taus = F.broadcast_to(taus[..., None], y.shape)
        I_delta = ((_unwrap_variable(t) - _unwrap_variable(y)) > 0).astype('f')
    eltwise_loss = (abs(taus - I_delta)
                    * F.huber_loss(y, t, delta=1.0, reduce='no'))
    return eltwise_loss


class IQN(dqn.DQN):
    """Implicit Quantile Networks.

    See https://arxiv.org/abs/1806.06923.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantile_thresholds_N = kwargs.get('quantile_thresholds_N', 8)
        self.quantile_thresholds_N_prime = kwargs.get(
            'quantile_thresholds_N_prime', 8)
        self.quantile_thresholds_K = kwargs.get('quantile_thresholds_K', 32)

    def _compute_target_values(self, exp_batch, gamma):
        """Compute a batch of target return distributions.

        Returns:
            chainer.Variable: (batch_size, N_prime).
        """
        batch_next_state = exp_batch['next_state']
        batch_size = batch_next_state.shape[0]
        taus_tilde = self.xp.random.uniform(
            0, 1, size=(batch_size, self.quantile_thresholds_K)).astype('f')

        target_next_tau2av = self.target_model(batch_next_state)
        greedy_actions = target_next_tau2av(taus_tilde).greedy_actions
        taus_prime = self.xp.random.uniform(
            0, 1,
            size=(batch_size, self.quantile_thresholds_N_prime)).astype('f')
        target_next_maxz = target_next_tau2av(
            taus_prime).evaluate_actions_as_quantiles(greedy_actions)

        batch_rewards = exp_batch['reward']
        batch_terminal = exp_batch['is_state_terminal']
        assert batch_rewards.shape == (batch_size,)
        assert batch_terminal.shape == (batch_size,)
        batch_rewards = F.broadcast_to(
            batch_rewards[..., None], target_next_maxz.shape)
        batch_terminal = F.broadcast_to(
            batch_terminal[..., None], target_next_maxz.shape)

        return (batch_rewards
                + gamma * (1.0 - batch_terminal) * target_next_maxz)

    def _compute_y_and_taus(self, exp_batch):
        """Compute a batch of predicted return distributions.

        Returns:
            chainer.Variable: Predicted return distributions.
                (batch_size, N).
        """

        batch_size = exp_batch['reward'].shape[0]

        # Compute Q-values for current states
        batch_state = exp_batch['state']

        # (batch_size, n_actions, n_atoms)
        tau2av = self.model(batch_state)
        taus = self.xp.random.uniform(
            0, 1, size=(batch_size, self.quantile_thresholds_N)).astype('f')
        av = tau2av(taus)
        batch_actions = exp_batch['action']
        y = av.evaluate_actions_as_quantiles(batch_actions)

        return y, taus

    def _compute_loss(self, exp_batch, gamma, errors_out=None):
        """Compute a loss.

        Returns:
            Returns:
                chainer.Variable: Scalar loss.
        """
        y, taus = self._compute_y_and_taus(exp_batch)
        with chainer.no_backprop_mode():
            t = self._compute_target_values(exp_batch, gamma)

        # Broadcast y and t to (batch_size, N, N_prime)
        y = F.expand_dims(y, axis=2)
        t = F.expand_dims(t, axis=1)
        y, t = F.broadcast(y, t)
        eltwise_loss = compute_eltwise_huber_quantile_loss(y, t, taus)

        if errors_out is not None:
            del errors_out[:]
            delta = F.mean(abs(eltwise_loss), axis=(1, 2))
            errors_out.extend(cuda.to_cpu(delta.data))

        if self.batch_accumulator == 'sum':
            return F.sum(F.mean(eltwise_loss, axis=2))
        else:
            return F.sum(F.mean(eltwise_loss, axis=(0, 2)))

    def act_and_train(self, obs, reward):

        with chainer.using_config('train', False), chainer.no_backprop_mode():
            taus_tilde = self.xp.random.uniform(
                0, 1, size=(1, self.quantile_thresholds_K)).astype('f')
            tau2av = self.model(
                self.batch_states([obs], self.xp, self.phi))
            action_value = tau2av(taus_tilde)
            q = float(action_value.max.data)
            greedy_action = cuda.to_cpu(action_value.greedy_actions.data)[0]

        # Update stats
        self.average_q *= self.average_q_decay
        self.average_q += (1 - self.average_q_decay) * q

        self.logger.debug('t:%s q:%s action_value:%s', self.t, q, action_value)

        action = self.explorer.select_action(
            self.t, lambda: greedy_action, action_value=action_value)
        self.t += 1

        # Update the target network
        if self.t % self.target_update_interval == 0:
            self.sync_target_network()

        if self.last_state is not None:
            assert self.last_action is not None
            # Add a transition to the replay buffer
            self.replay_buffer.append(
                state=self.last_state,
                action=self.last_action,
                reward=reward,
                next_state=obs,
                next_action=action,
                is_state_terminal=False)

        self.last_state = obs
        self.last_action = action

        self.replay_updater.update_if_necessary(self.t)

        self.logger.debug('t:%s r:%s a:%s', self.t, reward, action)

        return self.last_action

    def act(self, obs):
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            taus_tilde = self.xp.random.uniform(
                0, 1, size=(1, self.quantile_thresholds_K)).astype('f')
            tau2av = self.model(
                self.batch_states([obs], self.xp, self.phi))
            action_value = tau2av(taus_tilde)
            q = float(action_value.max.data)
            action = cuda.to_cpu(action_value.greedy_actions.data)[0]

        # Update stats
        self.average_q *= self.average_q_decay
        self.average_q += (1 - self.average_q_decay) * q

        self.logger.debug('t:%s q:%s action_value:%s', self.t, q, action_value)
        return action
