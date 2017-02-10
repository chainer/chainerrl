from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

import contextlib
import copy
from logging import getLogger

import chainer
from chainer import functions as F
import numpy as np

from chainerrl import agent
from chainerrl.misc import async
from chainerrl.misc import copy_param
from chainerrl.recurrent import Recurrent
from chainerrl.recurrent import RecurrentChainMixin
from chainerrl.recurrent import state_kept
from chainerrl.recurrent import state_reset

logger = getLogger(__name__)


class ACERSeparateModel(chainer.Chain, RecurrentChainMixin):
    """ACER model that consists of a separate policy and V-function.

    Args:
        pi (Policy): Policy.
        q (QFunction): Q-function.
    """

    def __init__(self, pi, q):
        super().__init__(pi=pi, q=q)

    def __call__(self, obs):
        pout = self.pi(obs)
        qout = self.q(obs)
        return pout, qout


class ACERSharedModel(chainer.Chain, RecurrentChainMixin):
    """ACER model where the policy and V-function share parameters.

    Args:
        shared (Link): Shared part. Nonlinearity must be included in it.
        pi (Policy): Policy that receives output of shared as input.
        q (QFunction): Q-function that receives output of shared as input.
    """

    def __init__(self, shared, pi, q):
        super().__init__(shared=shared, pi=pi, q=q)

    def __call__(self, obs):
        h = self.shared(obs)
        pout = self.pi(h)
        qout = self.q(h)
        return pout, qout


def compute_discrete_kl(p, q):
    """Compute KL divergence between two discrete distributions."""
    return F.sum(p.all_prob * (p.all_log_prob - q.all_log_prob), axis=1)


def compute_state_value_as_expected_action_value(action_value, action_distrib):
    """Compute state values as expected action values.

    Note that this does not backprop errrors because it is intended for use in
    computing target values.
    """
    return (action_distrib.all_prob.data *
            action_value.q_values.data).sum(axis=1)


@contextlib.contextmanager
def backprop_truncated(variable):
    backup = variable.creator
    variable.creator = None
    yield
    variable.creator = backup


class DiscreteACER(agent.AttributeSavingMixin, agent.AsyncAgent):
    """Discrete ACER (Actor-Critic with Experience Replay).

    See http://arxiv.org/abs/1611.01224

    Args:
        model (ACERModel): Model to train
        optimizer (chainer.Optimizer): optimizer used to train the model
        t_max (int): The model is updated after every t_max local steps
        gamma (float): Discount factor [0,1]
        beta (float): Weight coefficient for the entropy regularizaiton term.
        phi (callable): Feature extractor function
        pi_loss_coef (float): Weight coefficient for the loss of the policy
        Q_loss_coef (float): Weight coefficient for the loss of the value
            function
        normalize_loss_by_steps (bool): If set true, losses are normalized by
            the number of steps taken to accumulate the losses
        act_deterministically (bool): If set true, choose most probable actions
            in act method.
        average_entropy_decay (float): Decay rate of average entropy. Used only
            to record statistics.
        average_value_decay (float): Decay rate of average value. Used only
            to record statistics.
    """

    process_idx = None
    saved_attributes = ['model', 'optimizer']

    def __init__(self, model, optimizer, t_max, gamma, replay_buffer,
                 beta=1e-2,
                 process_idx=0, phi=lambda x: x,
                 pi_loss_coef=1.0, Q_loss_coef=0.5,
                 use_trust_region=True,
                 trust_region_alpha=0.99,
                 trust_region_c=10,
                 trust_region_delta=1,
                 disable_online_update=False,
                 n_times_replay=8,
                 replay_start_size=10 ** 4,
                 normalize_loss_by_steps=True,
                 act_deterministically=False,
                 eps_division=1e-6,
                 average_entropy_decay=0.999,
                 average_value_decay=0.999,
                 average_kl_decay=0.999):

        # Globally shared model
        self.shared_model = model

        # Globally shared average model used to compute trust regions
        self.shared_average_model = copy.deepcopy(self.shared_model)

        # Thread specific model
        self.model = copy.deepcopy(self.shared_model)
        async.assert_params_not_shared(self.shared_model, self.model)

        self.optimizer = optimizer

        self.replay_buffer = replay_buffer
        self.t_max = t_max
        self.gamma = gamma
        self.beta = beta
        self.phi = phi
        self.pi_loss_coef = pi_loss_coef
        self.Q_loss_coef = Q_loss_coef
        self.normalize_loss_by_steps = normalize_loss_by_steps
        self.act_deterministically = act_deterministically
        self.use_trust_region = use_trust_region
        self.trust_region_alpha = trust_region_alpha
        self.trust_region_c = trust_region_c
        self.trust_region_delta = trust_region_delta
        self.disable_online_update = disable_online_update
        self.n_times_replay = n_times_replay
        self.replay_start_size = replay_start_size
        self.eps_division = eps_division
        self.average_value_decay = average_value_decay
        self.average_entropy_decay = average_entropy_decay
        self.average_kl_decay = average_kl_decay

        self.t = 0
        self.last_state = None
        self.last_action = None
        # ACER won't use a explorer, but this arrtibute is referenced by
        # run_dqn
        self.explorer = None

        # Stats
        self.average_value = 0
        self.average_entropy = 0
        self.average_kl = 0

        self.init_history_data_for_online_update()

    def init_history_data_for_online_update(self):
        self.past_action_log_prob = {}
        self.past_states = {}
        self.past_actions = {}
        self.past_rewards = {}
        self.past_values = {}
        self.past_action_distrib = {}
        self.past_action_values = {}
        self.past_avg_action_distrib = {}
        self.t_start = self.t

    def sync_parameters(self):
        copy_param.copy_param(target_link=self.model,
                              source_link=self.shared_model)
        copy_param.soft_copy_param(target_link=self.shared_average_model,
                                   source_link=self.model,
                                   tau=1 - self.trust_region_alpha)

    @property
    def shared_attributes(self):
        return ('shared_model', 'shared_average_model', 'optimizer')

    def compute_one_step_pi_loss(self, advantage, action_distrib, log_prob,
                                 rho, rho_all, action_value, v,
                                 avg_action_distrib):
        # Compute gradients w.r.t statistics produced by the model

        # Compute g: a direction following policy gradients
        if rho is not None:
            # Off-policy
            g_loss = 0
            g_loss -= (min(self.trust_region_c, rho) *
                       log_prob * advantage)
            with chainer.no_backprop_mode():
                correction_weight = (
                    np.maximum(
                        1 - self.trust_region_c /
                        (rho_all + self.eps_division),
                        np.zeros_like(rho_all)) *
                    action_distrib.all_prob.data)
                correction_advantage = action_value.q_values.data - v
            g_loss -= F.sum(correction_weight *
                            action_distrib.all_log_prob *
                            correction_advantage, axis=1)
        else:
            # On-policy
            g_loss = -log_prob * advantage

        with backprop_truncated(action_distrib.logits):
            g_loss.backward()
        g = action_distrib.logits.grad[0]
        action_distrib.logits.grad = None

        # Compute k: a direction to increase KL div.
        if self.use_trust_region:
            neg_kl = -compute_discrete_kl(
                avg_action_distrib,
                action_distrib)
            self.average_kl += (
                (1 - self.average_kl_decay) *
                (-float(neg_kl.data) - self.average_kl))
            with backprop_truncated(action_distrib.logits):
                neg_kl.backward()
            k = action_distrib.logits.grad[0]
            action_distrib.logits.grad = None

        # Compute z: combination of g and k to keep small KL div.
        if self.use_trust_region and np.any(k):
            k_factor = max(0, ((np.dot(k, g) - self.trust_region_delta) /
                               (np.dot(k, k) + self.eps_division)))
            z = g - k_factor * k
        else:
            z = g
        pi_loss = 0
        # Backprop z
        pi_loss += F.sum(action_distrib.logits * z, axis=1)
        # Entropy is maximized
        pi_loss -= self.beta * action_distrib.entropy
        return pi_loss

    def update(self, t_start, t_stop, R, states, actions, rewards, values,
               action_values, action_log_probs,
               action_distribs, avg_action_distribs, rho=None, rho_all=None):

        pi_loss = 0
        Q_loss = 0
        Q_ret = R
        del R
        for i in reversed(range(t_start, t_stop)):
            r = rewards[i]
            v = values[i]
            log_prob = action_log_probs[i]
            assert isinstance(log_prob, chainer.Variable),\
                "log_prob must be backprop-able"
            action_distrib = action_distribs[i]
            avg_action_distrib = avg_action_distribs[i]
            ba = np.expand_dims(actions[i], 0)
            action_value = action_values[i]

            Q_ret = r + self.gamma * Q_ret

            with chainer.no_backprop_mode():
                advantage = Q_ret - v

            pi_loss += self.compute_one_step_pi_loss(
                advantage=advantage,
                action_distrib=action_distrib,
                log_prob=log_prob,
                rho=rho[i] if rho else None,
                rho_all=rho_all[i] if rho_all else None,
                action_value=action_value,
                v=v,
                avg_action_distrib=avg_action_distrib)

            # Accumulate gradients of value function
            Q = action_value.evaluate_actions(ba)
            assert isinstance(Q, chainer.Variable), "Q must be backprop-able"
            Q_loss += (Q_ret - Q) ** 2 / 2

            if self.process_idx == 0:
                logger.debug('t:%s s:%s v:%s Q:%s Q_ret:%s',
                             i, states[i].sum(), v, float(Q.data), Q_ret)

            if rho is not None:
                Q_ret = min(1, rho[i]) * (Q_ret - float(Q.data)) + v
            else:
                Q_ret = Q_ret - float(Q.data) + v

        pi_loss *= self.pi_loss_coef
        Q_loss *= self.Q_loss_coef

        if self.normalize_loss_by_steps:
            pi_loss /= t_stop - t_start
            Q_loss /= t_stop - t_start

        if self.process_idx == 0:
            logger.debug('pi_loss:%s Q_loss:%s', pi_loss.data, Q_loss.data)

        total_loss = pi_loss + F.reshape(Q_loss, pi_loss.data.shape)

        # Compute gradients using thread-specific model
        self.model.zerograds()
        total_loss.backward()
        # Copy the gradients to the globally shared model
        self.shared_model.zerograds()
        copy_param.copy_grad(
            target_link=self.shared_model, source_link=self.model)
        # Update the globally shared model
        if self.process_idx == 0:
            norm = self.optimizer.compute_grads_norm()
            logger.debug('grad norm:%s', norm)
        self.optimizer.update()

        self.sync_parameters()
        if isinstance(self.model, Recurrent):
            self.model.unchain_backward()

    def update_from_replay(self):

        if len(self.replay_buffer) < self.replay_start_size:
            return

        episode = self.replay_buffer.sample_episodes(1, self.t_max)[0]

        with state_reset(self.model):
            with state_reset(self.shared_average_model):
                rewards = {}
                states = {}
                actions = {}
                action_log_probs = {}
                action_distribs = {}
                avg_action_distribs = {}
                rho = {}
                rho_all = {}
                action_values = {}
                values = {}
                for t, transition in enumerate(episode):
                    s = self.phi(transition['state'])
                    a = transition['action']
                    ba = np.expand_dims(a, 0)
                    bs = np.expand_dims(s, 0)
                    action_distrib, action_value = self.model(bs)
                    v = compute_state_value_as_expected_action_value(
                        action_value, action_distrib)
                    with chainer.no_backprop_mode():
                        avg_action_distrib, _ = self.shared_average_model(bs)
                    states[t] = s
                    actions[t] = a
                    action_log_probs[t] = action_distrib.log_prob(ba)
                    values[t] = v
                    action_distribs[t] = action_distrib
                    avg_action_distribs[t] = avg_action_distrib
                    rewards[t] = transition['reward']
                    mu = transition['mu']
                    action_values[t] = action_value
                    rho[t] = (action_distrib.prob(ba).data /
                              (mu.prob(ba).data + self.eps_division))
                    rho_all[t] = (action_distrib.all_prob.data /
                                  (mu.all_prob.data + self.eps_division))
                last_transition = episode[-1]
                if last_transition['is_state_terminal']:
                    R = 0
                else:
                    with chainer.no_backprop_mode():
                        last_s = last_transition['next_state']
                        action_distrib, action_value = self.model(
                            np.expand_dims(self.phi(last_s), 0))
                        last_v = compute_state_value_as_expected_action_value(
                            action_value, action_distrib)
                    R = last_v
                return self.update(
                    R=R, t_start=0, t_stop=len(episode),
                    states=states, rewards=rewards,
                    actions=actions,
                    values=values,
                    action_log_probs=action_log_probs,
                    action_distribs=action_distribs,
                    avg_action_distribs=avg_action_distribs,
                    rho=rho,
                    rho_all=rho_all,
                    action_values=action_values)

    def update_on_policy(self, statevar):
        assert self.t_start < self.t

        if not self.disable_online_update:
            if statevar is None:
                R = 0
            else:
                with chainer.no_backprop_mode():
                    with state_kept(self.model):
                        action_distrib, action_value = self.model(statevar)
                        v = compute_state_value_as_expected_action_value(
                            action_value, action_distrib)
                R = v
            self.update(
                t_start=self.t_start, t_stop=self.t, R=R,
                states=self.past_states,
                actions=self.past_actions,
                rewards=self.past_rewards,
                values=self.past_values,
                action_values=self.past_action_values,
                action_log_probs=self.past_action_log_prob,
                action_distribs=self.past_action_distrib,
                avg_action_distribs=self.past_avg_action_distrib)

        self.init_history_data_for_online_update()

    def act_and_train(self, state, reward):

        statevar = np.expand_dims(self.phi(state), 0)

        self.past_rewards[self.t - 1] = reward

        if self.t - self.t_start == self.t_max:
            self.update_on_policy(statevar)
            for _ in range(self.n_times_replay):
                self.update_from_replay()

        self.past_states[self.t] = statevar
        action_distrib, action_value = self.model(statevar)
        self.past_action_values[self.t] = action_value
        action = action_distrib.sample()
        action.creator = None  # Do not backprop through sampled actions

        # Save values for a later update
        self.past_action_log_prob[self.t] = action_distrib.log_prob(action)
        v = compute_state_value_as_expected_action_value(
            action_value, action_distrib)
        self.past_values[self.t] = v
        self.past_action_distrib[self.t] = action_distrib
        with chainer.no_backprop_mode():
            avg_action_distrib, _ = self.shared_average_model(
                statevar)
        self.past_avg_action_distrib[self.t] = avg_action_distrib

        action = action.data[0]
        self.past_actions[self.t] = action

        self.t += 1

        if self.process_idx == 0:
            logger.debug('t:%s r:%s a:%s action_distrib:%s',
                         self.t, reward, action, action_distrib)
        # Update stats
        self.average_value += (
            (1 - self.average_value_decay) *
            (v - self.average_value))
        self.average_entropy += (
            (1 - self.average_entropy_decay) *
            (float(action_distrib.entropy.data[0]) - self.average_entropy))

        if self.last_state is not None:
            assert self.last_action is not None
            assert self.last_action_distrib is not None
            # Add a transition to the replay buffer
            self.replay_buffer.append(
                state=self.last_state,
                action=self.last_action,
                reward=reward,
                next_state=state,
                next_action=action,
                is_state_terminal=False,
                mu=self.last_action_distrib,
            )

        self.last_state = state
        self.last_action = action
        self.last_action_distrib = action_distrib.copy()

        return action

    def act(self, obs):
        # Use the process-local model for acting
        with chainer.no_backprop_mode():
            statevar = np.expand_dims(self.phi(obs), 0)
            action_distrib, _ = self.model(statevar)
            if self.act_deterministically:
                return action_distrib.most_probable.data[0]
            else:
                return action_distrib.sample().data[0]

    def stop_episode_and_train(self, state, reward, done=False):
        assert self.last_state is not None
        assert self.last_action is not None

        self.past_rewards[self.t - 1] = reward
        if done:
            self.update_on_policy(None)
        else:
            statevar = np.expand_dims(self.phi(state), 0)
            self.update_on_policy(statevar)
        for _ in range(self.n_times_replay):
            self.update_from_replay()

        if isinstance(self.model, Recurrent):
            self.model.reset_state()
            self.shared_average_model.reset_state()

        # Add a transition to the replay buffer
        self.replay_buffer.append(
            state=self.last_state,
            action=self.last_action,
            reward=reward,
            next_state=state,
            next_action=self.last_action,
            is_state_terminal=done,
            mu=self.last_action_distrib)
        self.replay_buffer.stop_current_episode()

        self.last_state = None
        self.last_action = None
        self.last_action_distrib = None

    def stop_episode(self):
        if isinstance(self.model, Recurrent):
            self.model.reset_state()
            self.shared_average_model.reset_state()

    def load(self, dirname):
        super().load(dirname)
        copy_param.copy_param(target_link=self.shared_model,
                              source_link=self.model)

    def get_statistics(self):
        return [
            ('average_value', self.average_value),
            ('average_entropy', self.average_entropy),
            ('average_kl', self.average_kl),
        ]
