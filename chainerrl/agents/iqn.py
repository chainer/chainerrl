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


def cosine_basis_functions(x, n_basis_functions=64):
    """Cosine basis functions used to embed quantile thresholds.

    Args:
        x (ndarray): Input.
        n_basis_functions (int): Number of cosine basis functions.

    Returns:
        ndarray: Embedding with shape of (x.shape + (n_basis_functions,)).
    """
    xp = chainer.cuda.get_array_module(x)
    # Equation (4) in the IQN paper has an error stating i=0,...,n-1.
    # Actually i=1,...,n is correct (personal communication)
    i_pi = xp.arange(1, n_basis_functions + 1, dtype=np.float32) * np.pi
    embedding = xp.cos(x[..., None] * i_pi)
    assert embedding.shape == x.shape + (n_basis_functions,)
    return embedding


class CosineBasisLinear(chainer.Chain):
    """Linear layer following cosine basis functions.

    Args:
        n_basis_functions (int): Number of cosine basis functions.
        out_size (int): Output size.
    """

    def __init__(self, n_basis_functions, out_size):
        super().__init__()
        with self.init_scope():
            self.linear = L.Linear(n_basis_functions, out_size)
        self.n_basis_functions = n_basis_functions
        self.out_size = out_size

    def __call__(self, x):
        """Evaluate.

        Args:
            x (ndarray): Input.

        Returns:
            chainer.Variable: Output with shape of (x.shape + (out_size,)).
        """
        h = cosine_basis_functions(x, self.n_basis_functions)
        h = F.reshape(h, (-1, self.n_basis_functions))
        out = self.linear(h)
        out = F.reshape(out, x.shape + (self.out_size,))
        return out


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
        QuantileDiscreteActionValue: Action values.
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
        assert psi_x.ndim == 2
        assert psi_x.shape[0] == batch_size
        hidden_size = psi_x.shape[1]

        def evaluate_with_quantile_thresholds(taus):
            assert taus.ndim == 2
            assert taus.shape[0] == batch_size
            n_taus = taus.shape[1]
            phi_taus = self.phi(taus)
            assert phi_taus.ndim == 3
            assert phi_taus.shape == (batch_size, n_taus, hidden_size)
            psi_x_b = F.broadcast_to(
                F.expand_dims(psi_x, axis=1), phi_taus.shape)
            h = psi_x_b * phi_taus
            h = F.reshape(h, (-1, hidden_size))
            assert h.shape == (batch_size * n_taus, hidden_size)
            h = self.f(h)
            assert h.ndim == 2
            assert h.shape[0] == batch_size * n_taus
            n_actions = h.shape[-1]
            h = F.reshape(h, (batch_size, n_taus, n_actions))
            return QuantileDiscreteActionValue(h)

        return evaluate_with_quantile_thresholds


def _unwrap_variable(x):
    if isinstance(x, chainer.Variable):
        return x.array
    else:
        return x


def compute_eltwise_huber_quantile_loss(y, t, taus, huber_loss_threshold=1.0):
    """Compute elementwise Huber losses for quantile regression.

    This is based on Algorithm 1 of https://arxiv.org/abs/1806.06923.

    This function assumes that, both of the two kinds of quantile thresholds,
    taus (used to compute y) and taus_prime (used to compute t) are iid samples
    from U([0,1]).

    Args:
        y (chainer.Variable): Quantile prediction from taus as a
            (batch_size, N)-shaped array.
        t (chainer.Variable or ndarray): Target values for quantile regression
            as a (batch_size, N_prime)-array.
        taus (ndarray): Quantile thresholds used to compute y as a
            (batch_size, N)-shaped array.
        huber_loss_threshold (float): Threshold of Huber loss. In the IQN
            paper, this is denoted by kappa.

    Returns:
        chainer.Variable: Loss (batch_size, N, N_prime)
    """
    assert y.shape == taus.shape
    # (batch_size, N) -> (batch_size, N, 1)
    y = F.expand_dims(y, axis=2)
    # (batch_size, N_prime) -> (batch_size, 1, N_prime)
    t = F.expand_dims(t, axis=1)
    # (batch_size, N) -> (batch_size, N, 1)
    taus = F.expand_dims(taus, axis=2)
    # Broadcast to (batch_size, N, N_prime)
    y, t, taus = F.broadcast(y, t, taus)
    I_delta = ((t.array - y.array) > 0).astype('f')
    eltwise_huber_loss = F.huber_loss(
        y, t, delta=huber_loss_threshold, reduce='no')
    eltwise_loss = abs(taus - I_delta) * eltwise_huber_loss
    return eltwise_loss


class IQN(dqn.DQN):
    """Implicit Quantile Networks.

    See https://arxiv.org/abs/1806.06923.

    Args:
        quantile_thresholds_N (int): Number of quantile thresholds used in
            quantile regression.
        quantile_thresholds_N_prime (int): Number of quantile thresholds used
            to sample from the return distribution at the next state.
        quantile_thresholds_K (int): Number of quantile thresholds used to
            compute greedy actions.

    For other arguments, see chainerrl.agents.DQN.
    """

    def __init__(self, *args, **kwargs):
        # N=N'=64 and K=32 were used in the IQN paper's experiments
        # (personal communication)
        self.quantile_thresholds_N = kwargs.pop('quantile_thresholds_N', 64)
        self.quantile_thresholds_N_prime = kwargs.pop(
            'quantile_thresholds_N_prime', 64)
        self.quantile_thresholds_K = kwargs.pop('quantile_thresholds_K', 32)
        super().__init__(*args, **kwargs)

    def _compute_target_values(self, exp_batch):
        """Compute a batch of target return distributions.

        Returns:
            chainer.Variable: (batch_size, N_prime).
        """
        batch_next_state = exp_batch['next_state']
        batch_size = len(exp_batch['reward'])
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
        batch_discount = exp_batch['discount']
        assert batch_rewards.shape == (batch_size,)
        assert batch_terminal.shape == (batch_size,)
        assert batch_discount.shape == (batch_size,)
        batch_rewards = F.broadcast_to(
            batch_rewards[..., None], target_next_maxz.shape)
        batch_terminal = F.broadcast_to(
            batch_terminal[..., None], target_next_maxz.shape)
        batch_discount = F.broadcast_to(
            batch_discount[..., None], target_next_maxz.shape)

        return (batch_rewards
                + batch_discount * (1.0 - batch_terminal) * target_next_maxz)

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

    def _compute_loss(self, exp_batch, errors_out=None):
        """Compute a loss.

        Returns:
            Returns:
                chainer.Variable: Scalar loss.
        """
        y, taus = self._compute_y_and_taus(exp_batch)
        with chainer.no_backprop_mode():
            t = self._compute_target_values(exp_batch)

        eltwise_loss = compute_eltwise_huber_quantile_loss(y, t, taus)

        if errors_out is not None:
            del errors_out[:]
            delta = F.mean(abs(eltwise_loss), axis=(1, 2))
            errors_out.extend(cuda.to_cpu(delta.array))

        if self.batch_accumulator == 'sum':
            # mean over N_prime, then sum over (batch_size, N)
            return F.sum(F.mean(eltwise_loss, axis=2))
        else:
            # mean over (batch_size, N_prime), then sum over N
            return F.sum(F.mean(eltwise_loss, axis=(0, 2)))

    def _compute_action_value(self, batch_obs):
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            taus_tilde = self.xp.random.uniform(
                0, 1,
                size=(len(batch_obs), self.quantile_thresholds_K)).astype('f')
            tau2av = self.model(
                self.batch_states(batch_obs, self.xp, self.phi))
            return tau2av(taus_tilde)

    def act_and_train(self, obs, reward):
        action_value = self._compute_action_value([obs])
        greedy_action = cuda.to_cpu(action_value.greedy_actions.array)[0]
        q = float(action_value.max.array)

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
        action_value = self._compute_action_value([obs])
        action = cuda.to_cpu(action_value.greedy_actions.array)[0]
        q = float(action_value.max.array)

        # Update stats
        self.average_q *= self.average_q_decay
        self.average_q += (1 - self.average_q_decay) * q

        self.logger.debug('t:%s q:%s action_value:%s', self.t, q, action_value)
        return action

    def batch_act_and_train(self, batch_obs):
        batch_av = self._compute_action_value(batch_obs)
        batch_maxq = batch_av.max.array
        batch_argmax = cuda.to_cpu(batch_av.greedy_actions.array)
        batch_action = [
            self.explorer.select_action(
                self.t, lambda: batch_argmax[i],
                action_value=batch_av[i:i + 1],
            )
            for i in range(len(batch_obs))]
        self.batch_last_obs = list(batch_obs)
        self.batch_last_action = list(batch_action)

        # Update stats
        self.average_q *= self.average_q_decay
        self.average_q += (1 - self.average_q_decay) * float(batch_maxq.mean())

        return batch_action

    def batch_act(self, batch_obs):
        batch_av = self._compute_action_value(batch_obs)
        batch_argmax = cuda.to_cpu(batch_av.greedy_actions.array)
        return batch_argmax
