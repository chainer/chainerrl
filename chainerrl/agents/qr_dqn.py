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
from chainerrl.functions import quantile_loss


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

    def __init__(self, q_quantiles):
        assert isinstance(q_quantiles, chainer.Variable)
        assert q_quantiles.ndim == 3
        self.xp = cuda.get_array_module(q_quantiles.data)
        self.q_quantiles = q_quantiles
        self.n_actions = q_quantiles.data.shape[1]
        self.n_diracs = q_quantiles.data.shape[2]

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

    def __repr__(self):
        return '_ActionQuantile greedy_actions:{} q_quantiles:\n{}'.format(
            self.greedy_actions.data,
            self.q_quantiles.data)

    @property
    def std_greedy(self):
        quantiles = self.evaluate_actions(self.greedy_actions)
        mean = F.mean(quantiles, axis=1, keepdims=True)
        std = F.sqrt(F.mean(
            F.square(
                quantiles - F.broadcast_to(mean, quantiles.shape)),
            axis=1))
        return std


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
        lossfun (str): 'l1' or 'huber' (QR-DQN-0 or QR-DQN-1 on the paper)
        args of DQN

    See: https://arxiv.org/abs/1710.10044
    """

    def __init__(self, *args, **kwargs):
        lossfun = kwargs.pop('lossfun', 'l1')
        assert lossfun in ['l1', 'huber']
        self._lossfun = {
            'l1': quantile_loss,
            'huber': quantile_huber_loss_Dabney,
        }[lossfun]
        super().__init__(*args, **kwargs)
        self.average_qstd = 0

    def get_statistics(self):
        return super().get_statistics() + [
            ('average_qstd', self.average_qstd),
        ]

    def _compute_loss(self, exp_batch, gamma, errors_out=None):
        xp = self.xp

        y, t = self._compute_y_and_t(exp_batch, gamma)
        _, n_diracs = y.shape
        assert y.shape[0] == t.shape[0]

        # broadcast to (batch, t_n_dirac, y_n_dirac)
        y, t = F.broadcast(y[:, None, :], t[:, :, None])

        tau_hat = (xp.arange(n_diracs).astype(y.dtype) + 0.5) / n_diracs
        tau_hat = F.broadcast_to(tau_hat, y.shape)

        loss = self._lossfun(y, t, tau_hat)
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

    def act(self, state):
        with chainer.using_config('train', False):
            with chainer.no_backprop_mode():
                action_value = self.model(
                    self.batch_states([state], self.xp, self.phi))
                q = float(action_value.max.data)
                action = cuda.to_cpu(action_value.greedy_actions.data)[0]

        # Update stats
        self.average_q *= self.average_q_decay
        self.average_q += (1 - self.average_q_decay) * q

        self.average_qstd *= self.average_q_decay
        self.average_qstd += (1 - self.average_q_decay) * cuda.to_cpu(
            action_value.std_greedy.data)[0]

        self.logger.debug('t:%s q:%s action_value:%s', self.t, q, action_value)
        return action

    def act_and_train(self, state, reward):

        with chainer.using_config('train', False):
            with chainer.no_backprop_mode():
                action_value = self.model(
                    self.batch_states([state], self.xp, self.phi))
                q = float(action_value.max.data)
                greedy_action = cuda.to_cpu(action_value.greedy_actions.data)[
                    0]

        # Update stats
        self.average_q *= self.average_q_decay
        self.average_q += (1 - self.average_q_decay) * q

        self.average_qstd *= self.average_q_decay
        self.average_qstd += (1 - self.average_q_decay) * cuda.to_cpu(
            action_value.std_greedy.data)[0]

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
                next_state=state,
                next_action=action,
                is_state_terminal=False)

        self.last_state = state
        self.last_action = action

        self.replay_updater.update_if_necessary(self.t)

        self.logger.debug('t:%s r:%s a:%s', self.t, reward, action)

        return self.last_action
