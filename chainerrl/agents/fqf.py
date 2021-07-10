from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA
import collections

import chainer
from chainer import cuda
import chainer.functions as F
import numpy as np

from chainerrl.action_value import QuantileDiscreteActionValue
from chainerrl.agents import dqn
from chainerrl.links import StatelessRecurrentChainList
from chainerrl.agents import iqn


def _mean_or_nan(xs):
    """Return its mean a non-empty sequence, numpy.nan for a empty one."""
    return np.mean(xs) if xs else np.nan


def _evaluate_psi_x_with_quantile_thresholds(
        psi_x, phi, f, taus, weights=None):
    assert psi_x.ndim == 2
    batch_size, hidden_size = psi_x.shape
    assert taus.ndim == 2
    assert taus.shape[0] == batch_size
    n_taus = taus.shape[1]
    phi_taus = phi(taus)
    assert phi_taus.ndim == 3
    assert phi_taus.shape == (batch_size, n_taus, hidden_size)
    psi_x_b = F.broadcast_to(
        F.expand_dims(psi_x, axis=1), phi_taus.shape)
    h = psi_x_b * phi_taus
    h = F.reshape(h, (-1, hidden_size))
    assert h.shape == (batch_size * n_taus, hidden_size)
    h = f(h)
    assert h.ndim == 2
    assert h.shape[0] == batch_size * n_taus
    n_actions = h.shape[-1]
    h = F.reshape(h, (batch_size, n_taus, n_actions))
    return QuantileDiscreteActionValue(h, weights=weights)


def _compute_taus_hat_and_weights(taus):
    batch_size, _ = taus.shape
    xp = chainer.cuda.get_array_module(taus)
    _assert_taus(taus)
    # shifted_tau: [0, tau_0, tau_1, ..., tau_{N-2}]
    shifted_tau = xp.concatenate(
        [xp.zeros((batch_size, 1), dtype=taus.dtype), taus[:, :-1]],
        axis=1,
    )
    weights = taus - shifted_tau
    taus_hat = (taus + shifted_tau) / 2
    assert taus_hat.shape == taus.shape
    assert weights.shape == weights.shape
    return taus_hat, weights


class FQQFunction(chainer.Chain):

    """Fully-parameterized Quantile Q-function.

    Args:
        psi (chainer.Link): Callable link
            (batch_size, obs_size) -> (batch_size, hidden_size).
        phi (chainer.Link): Callable link
            (batch_size, n_taus) -> (batch_size, n_taus, hidden_size).
        f (chainer.Link): Callable link
            (batch_size * n_taus, hidden_size)
            -> (batch_size * n_taus, n_actions).
        proposal_net (chainer.Link): Callable link
            (batch_size, hidden_size) -> (batch_size, n_taus)
    """

    def __init__(self, psi, phi, f, proposal_net):
        super().__init__()
        with self.init_scope():
            self.psi = psi
            self.phi = phi
            self.f = f
            self.proposal_net = proposal_net

    def __call__(self, x, taus=None, with_tau_quantiles=False):
        """Evaluate given observations.

        Args:
            x (ndarray): Batch of observations.
            taus (None or ndarray): Taus (Quantile thresholds). If set to None,
                the proposal net is used to compute taus.
            with_tau_quantiles (bool): If set to True, results with taus are
                returned besides ones with taus hat.

        Returns:
            QuantileDiscreteActionValue: ActionValue based on tau hat.
            ndarray: Tau hat.
            QuantileDiscreteActionValue: ActionValue based on tau.
                Returned only when with_tau_quantiles=True.
            ndarray: Tau. Returned only when with_tau_quantiles=True.
        """
        batch_size = x.shape[0]
        psi_x = self.psi(x)
        assert psi_x.ndim == 2
        assert psi_x.shape[0] == batch_size
        if taus is None:
            # Make sure errors of the proposal net do not backprop to psi
            taus = F.cumsum(
                F.softmax(self.proposal_net(psi_x.array), axis=1), axis=1)
        _assert_taus(taus)

        # Quantiles based on tau hat, used to compute Q-values
        taus_hat, weights = _compute_taus_hat_and_weights(
            _unwrap_variable(taus))
        tau_hat_av = _evaluate_psi_x_with_quantile_thresholds(
            psi_x, self.phi, self.f, taus_hat, weights=weights)

        if with_tau_quantiles:
            # Quantiles based on tau, used to update the proposal net.
            # Since we don't compute Q-values based on tau, we don't need to
            # specify weights here.
            tau_av = _evaluate_psi_x_with_quantile_thresholds(
                psi_x, self.phi, self.f, _unwrap_variable(taus))
            return tau_hat_av, taus_hat, tau_av, taus
        else:
            return tau_hat_av, taus_hat


class StatelessRecurrentFQQFunction(
        StatelessRecurrentChainList):

    """Recurrent Fully-parameterized Quantile Q-function.

    Args:
        psi (chainer.Link): Link that implements
            `chainerrl.links.StatelessRecurrent`.
            (batch_size, obs_size) -> (batch_size, hidden_size).
        phi (chainer.Link): Callable link
            (batch_size, n_taus) -> (batch_size, n_taus, hidden_size).
        f (chainer.Link): Callable link
            (batch_size * n_taus, hidden_size)
            -> (batch_size * n_taus, n_actions).
        proposal_net (chainer.Link): Callable link
            (batch_size, hidden_size) -> (batch_size, n_taus)
    """

    def __init__(self, psi, phi, f, proposal_net):
        super().__init__(psi, phi, f, proposal_net)
        self.psi = psi
        self.phi = phi
        self.f = f
        self.proposal_net = proposal_net

    def n_step_forward(
            self,
            x,
            recurrent_state,
            output_mode,
            taus=None,
            with_tau_quantiles=False,
    ):
        """Evaluate given observations.

        Args:
            x (ndarray): Batch of observations.
        Returns:
            callable: (batch_size, taus) -> (batch_size, taus, n_actions)
        """
        assert output_mode == 'concat'
        if recurrent_state is not None:
            recurrent_state, = recurrent_state
        psi_x, recurrent_state = self.psi.n_step_forward(
            x, recurrent_state, output_mode='concat')
        assert psi_x.ndim == 2

        if taus is None:
            # Make sure errors of the proposal net do not backprop to psi
            taus = F.cumsum(
                F.softmax(self.proposal_net(psi_x.array), axis=1), axis=1)
        _assert_taus(taus)

        # Quantiles based on tau hat, used to compute Q-values
        taus_hat, weights = _compute_taus_hat_and_weights(
            _unwrap_variable(taus))
        tau_hat_av = _evaluate_psi_x_with_quantile_thresholds(
            psi_x, self.phi, self.f, taus_hat, weights=weights)

        if with_tau_quantiles:
            # Quantiles based on tau, used to update the proposal net
            # Since we don't compute Q-values based on tau, we don't need to
            # specify weights here.
            tau_av = _evaluate_psi_x_with_quantile_thresholds(
                psi_x, self.phi, self.f, _unwrap_variable(taus))
            return (tau_hat_av, taus_hat, tau_av, taus), (recurrent_state,)
        else:
            return (tau_hat_av, taus_hat), (recurrent_state,)


def _unwrap_variable(x):
    if isinstance(x, chainer.Variable):
        return x.array
    else:
        return x


def _assert_taus(taus):
    xp = chainer.cuda.get_array_module(taus)
    taus = _unwrap_variable(taus)
    # all the elements must be less than or equal to 1
    assert xp.all(taus <= 1 + 1e-6), taus
    # the last element must be 1
    assert xp.allclose(taus[:, -1], xp.ones(len(taus))), taus


def _restore_probs_from_taus(taus):
    _assert_taus(taus)
    taus = _unwrap_variable(taus)
    xp = chainer.cuda.get_array_module(taus)
    probs = taus.copy()
    probs[:, 1:] -= taus[:, :-1]
    assert xp.allclose(probs.sum(axis=1), xp.ones(len(taus)))
    return probs


def _mean_entropy(probs):
    assert probs.ndim == 2
    xp = chainer.cuda.get_array_module(probs)
    return -float(xp.mean(xp.sum(probs * xp.log(probs + 1e-8), axis=1)))


class FQF(dqn.DQN):

    """Fully-parameterized Quantile Function (FQF) algorithm.

    See http://arxiv.org/abs/1911.02140.

    Args:
        model (FQQFunction): Q-function link to train.

    For other arguments, see chainerrl.agents.DQN.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tau_grad_norm_record = collections.deque(maxlen=100)
        self.proposal_entropy_record = collections.deque(maxlen=100)

    def _compute_target_values(self, exp_batch, taus):
        """Compute a batch of target return distributions.

        Returns:
            chainer.Variable: (batch_size, N_prime).
        """
        batch_next_state = exp_batch['next_state']
        batch_size = len(exp_batch['reward'])

        if self.recurrent:
            (target_next_av, _), _ = self.target_model.n_step_forward(
                batch_next_state,
                exp_batch['next_recurrent_state'],
                output_mode='concat',
                taus=taus,
            )
        else:
            target_next_av, _ = self.target_model(batch_next_state, taus=taus)
        greedy_actions = target_next_av.greedy_actions
        target_next_maxz = target_next_av.evaluate_actions_as_quantiles(
            greedy_actions)

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

        return (batch_rewards +
                batch_discount * (1.0 - batch_terminal) * target_next_maxz)

    def _compute_predictions(self, exp_batch):
        """Compute a batch of predicted return distributions.

        Returns:
            chainer.Variable: Predicted return distributions.
                (batch_size, N).
        """

        # Compute Q-values for current states
        batch_state = exp_batch['state']

        # (batch_size, n_actions, n_atoms)
        if self.recurrent:
            (tau_hat_av, taus_hat, tau_av, taus), _ = self.model.n_step_forward(  # NOQA
                batch_state,
                exp_batch['recurrent_state'],
                output_mode='concat',
                with_tau_quantiles=True,
            )
        else:
            tau_hat_av, taus_hat, tau_av, taus = self.model(
                batch_state, with_tau_quantiles=True)
        batch_actions = exp_batch['action']
        taus_hat_quantiles = tau_hat_av.evaluate_actions_as_quantiles(
            batch_actions)
        tau_quantiles = tau_av.evaluate_actions_as_quantiles(batch_actions)

        return taus_hat_quantiles, taus_hat, tau_quantiles, taus

    def _compute_loss(self, exp_batch, errors_out=None):
        """Compute a loss.

        Returns:
            Returns:
                chainer.Variable: Scalar loss.
        """
        tau_hat_quantiles, taus_hat, tau_quantiles, taus =\
            self._compute_predictions(exp_batch)
        with chainer.no_backprop_mode():
            target_quantiles = self._compute_target_values(
                exp_batch, taus)

        eltwise_loss = iqn.compute_eltwise_huber_quantile_loss(
            tau_hat_quantiles, target_quantiles, taus_hat)

        tau_grad = (2 * tau_quantiles[:, :-1]
                    - tau_hat_quantiles[:, :-1]
                    - tau_hat_quantiles[:, 1:]).array
        xp = chainer.cuda.get_array_module(tau_grad)
        proposal_loss = F.mean(F.sum(tau_grad * taus[:, :-1], axis=1))

        # Record norm of \partial W_1 / \partial \tau
        tau_grad_norm = xp.mean(xp.linalg.norm(tau_grad, axis=1))
        self.tau_grad_norm_record.append(float(tau_grad_norm))

        # Record entropy of proposals
        self.proposal_entropy_record.append(
            _mean_entropy(_restore_probs_from_taus(taus)))

        if errors_out is not None:
            del errors_out[:]
            delta = F.mean(eltwise_loss, axis=(1, 2))
            errors_out.extend(cuda.to_cpu(delta.array))

        if 'weights' in exp_batch:
            return proposal_loss + iqn.compute_weighted_value_loss(
                eltwise_loss, exp_batch['weights'],
                batch_accumulator=self.batch_accumulator)
        else:
            return proposal_loss + iqn.compute_value_loss(
                eltwise_loss, batch_accumulator=self.batch_accumulator)

    def _evaluate_model_and_update_recurrent_states(self, batch_obs, test):
        batch_xs = self.batch_states(batch_obs, self.xp, self.phi)
        if self.recurrent:
            if test:
                (av, _), self.test_recurrent_states = self.model(
                    batch_xs, recurrent_state=self.test_recurrent_states)
            else:
                self.train_prev_recurrent_states = self.train_recurrent_states
                (av, _), self.train_recurrent_states = self.model(
                    batch_xs, recurrent_state=self.train_recurrent_states)
        else:
            av, _ = self.model(batch_xs)
        return av

    def get_statistics(self):
        return super().get_statistics() + [
            ('average_tau_grad_norm', _mean_or_nan(self.tau_grad_norm_record)),
            ('average_proposal_entropy', _mean_or_nan(
                self.proposal_entropy_record)),
        ]
