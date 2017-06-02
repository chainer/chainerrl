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

from chainerrl.action_value import SingleActionValue
from chainerrl import agent
from chainerrl import distribution
from chainerrl import links
from chainerrl.misc import async
from chainerrl.misc import copy_param
from chainerrl.recurrent import Recurrent
from chainerrl.recurrent import RecurrentChainMixin
from chainerrl.recurrent import state_kept
from chainerrl.recurrent import state_reset


def compute_importance(pi, mu, x):
    return np.nan_to_num(float(pi.prob(x).data) / float(mu.prob(x).data))


def compute_full_importance(pi, mu):
    pimu = pi.all_prob.data / mu.all_prob.data
    # NaN occurs when inf/inf or 0/0
    pimu[np.isnan(pimu)] = 1
    pimu = np.nan_to_num(pimu)
    return pimu


def compute_policy_gradient_full_correction(
        action_distrib, action_distrib_mu, action_value, v,
        truncation_threshold):
    """Compute off-policy bias correction term wrt all actions."""
    assert truncation_threshold is not None
    assert np.isscalar(v)
    with chainer.no_backprop_mode():
        rho_all_inv = compute_full_importance(action_distrib_mu,
                                              action_distrib)
        correction_weight = (
            np.maximum(1 - truncation_threshold * rho_all_inv,
                       np.zeros_like(rho_all_inv)) *
            action_distrib.all_prob.data[0])
        correction_advantage = action_value.q_values.data[0] - v
    return -F.sum(correction_weight *
                  action_distrib.all_log_prob *
                  correction_advantage, axis=1)


def compute_policy_gradient_sample_correction(
        action_distrib, action_distrib_mu, action_value, v,
        truncation_threshold):
    """Compute off-policy bias correction term wrt a sampled action."""
    assert np.isscalar(v)
    assert truncation_threshold is not None
    with chainer.no_backprop_mode():
        sample_action = action_distrib.sample().data
        rho_dash_inv = compute_importance(
            action_distrib_mu, action_distrib, sample_action)
        if (truncation_threshold > 0 and
                rho_dash_inv >= 1 / truncation_threshold):
            return chainer.Variable(np.asarray([0], dtype=np.float32))
        correction_weight = max(0, 1 - truncation_threshold * rho_dash_inv)
        assert correction_weight <= 1
        q = float(action_value.evaluate_actions(sample_action).data[0])
        correction_advantage = q - v
    return -(correction_weight *
             action_distrib.log_prob(sample_action) *
             correction_advantage)


def compute_policy_gradient_loss(action, advantage, action_distrib,
                                 action_distrib_mu, action_value, v,
                                 truncation_threshold):
    """Compute policy gradient loss with off-policy bias correction."""
    assert np.isscalar(advantage)
    assert np.isscalar(v)
    log_prob = action_distrib.log_prob(action)
    if action_distrib_mu is not None:
        # Off-policy
        rho = compute_importance(
            action_distrib, action_distrib_mu, action)
        g_loss = 0
        if truncation_threshold is None:
            g_loss -= rho * log_prob * advantage
        else:
            # Truncated off-policy policy gradient term
            g_loss -= min(truncation_threshold, rho) * log_prob * advantage
            # Bias correction term
            if isinstance(action_distrib,
                          distribution.CategoricalDistribution):
                g_loss += compute_policy_gradient_full_correction(
                    action_distrib=action_distrib,
                    action_distrib_mu=action_distrib_mu,
                    action_value=action_value,
                    v=v,
                    truncation_threshold=truncation_threshold)
            else:
                g_loss += compute_policy_gradient_sample_correction(
                    action_distrib=action_distrib,
                    action_distrib_mu=action_distrib_mu,
                    action_value=action_value,
                    v=v,
                    truncation_threshold=truncation_threshold)
    else:
        # On-policy
        g_loss = -log_prob * advantage
    return g_loss


class ACERSeparateModel(chainer.Chain, RecurrentChainMixin):
    """ACER model that consists of a separate policy and V-function.

    Args:
        pi (Policy): Policy.
        q (QFunction): Q-function.
    """

    def __init__(self, pi, q):
        super().__init__(pi=pi, q=q)

    def __call__(self, obs):
        action_distrib = self.pi(obs)
        action_value = self.q(obs)
        v = F.sum(action_distrib.all_prob *
                  action_value.q_values, axis=1)
        return action_distrib, action_value, v


class ACERSDNSeparateModel(chainer.Chain, RecurrentChainMixin):
    """ACER model that consists of a separate policy and V-function.

    Args:
        pi (Policy): Policy.
        v (VFunction): V-function.
        adv (StateActionQFunction): Advantage function.
    """

    def __init__(self, pi, v, adv, n=5):
        super().__init__(pi=pi, v=v, adv=adv)
        self.n = n

    def __call__(self, obs):
        action_distrib = self.pi(obs)
        v = self.v(obs)

        def evaluator(action):
            adv_mean = sum(self.adv(obs, action_distrib.sample().data)
                           for _ in range(self.n)) / self.n
            return v + self.adv(obs, action) - adv_mean

        action_value = SingleActionValue(evaluator)

        return action_distrib, action_value, v


class ACERSDNSharedModel(links.Sequence, RecurrentChainMixin):
    """ACER model where the policy and V-function share parameters.

    Args:
        shared (Link): Shared part. Nonlinearity must be included in it.
        pi (Policy): Policy that receives output of shared as input.
        q (QFunction): Q-function that receives output of shared as input.
    """

    def __init__(self, shared, pi, v, adv):
        super().__init__(shared, ACERSDNSeparateModel(pi, v, adv))


class ACERSharedModel(links.Sequence, RecurrentChainMixin):
    """ACER model where the policy and V-function share parameters.

    Args:
        shared (Link): Shared part. Nonlinearity must be included in it.
        pi (Policy): Policy that receives output of shared as input.
        q (QFunction): Q-function that receives output of shared as input.
    """

    def __init__(self, shared, pi, q):
        super().__init__(shared, ACERSeparateModel(pi, q))


@contextlib.contextmanager
def backprop_truncated(*variables):
    backup = [v.creator for v in variables]
    for v in variables:
        v.unchain()
    yield
    for v, backup_creator in zip(variables, backup):
        v.set_creator(backup_creator)


def compute_loss_with_kl_constraint(distrib, another_distrib, original_loss,
                                    delta):
    """Compute loss considering a KL constraint.

    Args:
        distrib (Distribution): Distribution to optimize
        another_distrib (Distribution): Distribution used to compute KL
        original_loss (chainer.Variable): Loss to minimize
        delta (float): Minimum KL difference
    Returns:
        loss (chainer.Variable)
    """
    # Compute g: a direction to minimize the original loss
    with backprop_truncated(*distrib.params):
        original_loss.backward()
    g = [p.grad[0] for p in distrib.params]
    for p in distrib.params:
        p.cleargrad()

    # Compute k: a direction to increase KL div.
    kl = another_distrib.kl(distrib)
    with backprop_truncated(*distrib.params):
        (-kl).backward()
    k = [p.grad[0] for p in distrib.params]
    for p in distrib.params:
        p.cleargrad()

    # Compute z: combination of g and k to keep small KL div.
    kg_dot = sum(np.dot(kp.ravel(), gp.ravel())
                 for kp, gp in zip(k, g))
    kk_dot = sum(np.dot(kp.ravel(), kp.ravel()) for kp in k)
    if kk_dot > 0:
        k_factor = max(0, ((kg_dot - delta) / kk_dot))
    else:
        k_factor = 0
    z = [gp - k_factor * kp for kp, gp in zip(k, g)]
    loss = 0
    for p, zp in zip(distrib.params, z):
        loss += F.sum(p * zp, axis=1)
    return loss, float(kl.data)


class ACER(agent.AttributeSavingMixin, agent.AsyncAgent):
    """ACER (Actor-Critic with Experience Replay).

    See http://arxiv.org/abs/1611.01224

    Args:
        model (ACERModel): Model to train. It must be a callable that accepts
            observations as input and return three values: action distributions
            (Distribution), Q values (ActionValue) and state values
            (chainer.Variable).
        optimizer (chainer.Optimizer): optimizer used to train the model
        t_max (int): The model is updated after every t_max local steps
        gamma (float): Discount factor [0,1]
        replay_buffer (EpisodicReplayBuffer): Replay buffer to use. If set
            None, this agent won't use experience replay.
        beta (float): Weight coefficient for the entropy regularizaiton term.
        phi (callable): Feature extractor function
        pi_loss_coef (float): Weight coefficient for the loss of the policy
        Q_loss_coef (float): Weight coefficient for the loss of the value
            function
        use_trust_region (bool): If set true, use efficient TRPO.
        trust_region_alpha (float): Decay rate of the average model used for
            efficient TRPO.
        trust_region_delta (float): Threshold used for efficient TRPO.
        truncation_threshold (float or None): Threshold used to truncate larger
            importance weights. If set None, importance weights are not
            truncated.
        disable_online_update (bool): If set true, disable online on-policy
            update and rely only on experience replay.
        n_times_replay (int): Number of times experience replay is repeated per
            one time of online update.
        replay_start_size (int): Experience replay is disabled if the number of
            transitions in the replay buffer is lower than this value.
        normalize_loss_by_steps (bool): If set true, losses are normalized by
            the number of steps taken to accumulate the losses
        act_deterministically (bool): If set true, choose most probable actions
            in act method.
        use_Q_opc (bool): If set true, use Q_opc, a Q-value estimate without
            importance sampling, is used to compute advantage values for policy
            gradients. The original paper recommend to use in case of
            continuous action.
        average_entropy_decay (float): Decay rate of average entropy. Used only
            to record statistics.
        average_value_decay (float): Decay rate of average value. Used only
            to record statistics.
        average_kl_decay (float): Decay rate of kl value. Used only to record
            statistics.
    """

    process_idx = None
    saved_attributes = ['model', 'optimizer']

    def __init__(self, model, optimizer, t_max, gamma, replay_buffer,
                 beta=1e-2,
                 phi=lambda x: x,
                 pi_loss_coef=1.0,
                 Q_loss_coef=0.5,
                 use_trust_region=True,
                 trust_region_alpha=0.99,
                 trust_region_delta=1,
                 truncation_threshold=10,
                 disable_online_update=False,
                 n_times_replay=8,
                 replay_start_size=10 ** 4,
                 normalize_loss_by_steps=True,
                 act_deterministically=False,
                 use_Q_opc=False,
                 average_entropy_decay=0.999,
                 average_value_decay=0.999,
                 average_kl_decay=0.999,
                 logger=None):

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
        self.truncation_threshold = truncation_threshold
        self.trust_region_delta = trust_region_delta
        self.disable_online_update = disable_online_update
        self.n_times_replay = n_times_replay
        self.use_Q_opc = use_Q_opc
        self.replay_start_size = replay_start_size
        self.average_value_decay = average_value_decay
        self.average_entropy_decay = average_entropy_decay
        self.average_kl_decay = average_kl_decay
        self.logger = logger if logger else getLogger(__name__)

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

    def compute_one_step_pi_loss(self, action, advantage, action_distrib,
                                 action_distrib_mu, action_value, v,
                                 avg_action_distrib):
        assert np.isscalar(advantage)
        assert np.isscalar(v)

        g_loss = compute_policy_gradient_loss(
            action=action,
            advantage=advantage,
            action_distrib=action_distrib,
            action_distrib_mu=action_distrib_mu,
            action_value=action_value,
            v=v,
            truncation_threshold=self.truncation_threshold)

        if self.use_trust_region:
            pi_loss, kl = compute_loss_with_kl_constraint(
                action_distrib, avg_action_distrib, g_loss,
                delta=self.trust_region_delta)
            self.average_kl += (
                (1 - self.average_kl_decay) * (kl - self.average_kl))
        else:
            pi_loss = g_loss

        # Entropy is maximized
        pi_loss -= self.beta * action_distrib.entropy
        return pi_loss

    def compute_loss(
            self, t_start, t_stop, R, states, actions, rewards, values,
            action_values, action_distribs, action_distribs_mu,
            avg_action_distribs):

        assert np.isscalar(R)
        pi_loss = 0
        Q_loss = 0
        Q_ret = R
        Q_opc = R
        discrete = isinstance(action_distribs[t_start],
                              distribution.CategoricalDistribution)
        del R
        for i in reversed(range(t_start, t_stop)):
            r = rewards[i]
            v = values[i]
            action_distrib = action_distribs[i]
            action_distrib_mu = (action_distribs_mu[i]
                                 if action_distribs_mu else None)
            avg_action_distrib = avg_action_distribs[i]
            action_value = action_values[i]
            ba = np.expand_dims(actions[i], 0)
            if action_distrib_mu is not None:
                # Off-policy
                rho = compute_importance(action_distrib, action_distrib_mu, ba)
            else:
                # On-policy
                rho = 1

            Q_ret = r + self.gamma * Q_ret
            Q_opc = r + self.gamma * Q_opc

            assert np.isscalar(Q_ret)
            assert np.isscalar(Q_opc)
            if self.use_Q_opc:
                advantage = Q_opc - float(v.data)
            else:
                advantage = Q_ret - float(v.data)
            pi_loss += self.compute_one_step_pi_loss(
                action=ba,
                advantage=advantage,
                action_distrib=action_distrib,
                action_distrib_mu=action_distrib_mu,
                action_value=action_value,
                v=float(v.data),
                avg_action_distrib=avg_action_distrib)

            # Accumulate gradients of value function
            Q = action_value.evaluate_actions(ba)
            assert isinstance(Q, chainer.Variable), "Q must be backprop-able"
            Q_loss += (Q_ret - Q) ** 2 / 2

            if not discrete:
                assert isinstance(v, chainer.Variable), \
                    "v must be backprop-able"
                v_target = (min(1, rho) * (Q_ret - float(Q.data)) +
                            float(v.data))
                Q_loss += (v_target - v) ** 2 / 2

            if self.process_idx == 0:
                self.logger.debug(
                    't:%s v:%s Q:%s Q_ret:%s Q_opc:%s',
                    i, float(v.data), float(Q.data), Q_ret, Q_opc)

            if discrete:
                c = min(1, rho)
            else:
                c = min(1, rho ** (1 / ba.size))
            Q_ret = c * (Q_ret - float(Q.data)) + float(v.data)
            Q_opc = Q_opc - float(Q.data) + float(v.data)

        pi_loss *= self.pi_loss_coef
        Q_loss *= self.Q_loss_coef

        if self.normalize_loss_by_steps:
            pi_loss /= t_stop - t_start
            Q_loss /= t_stop - t_start

        if self.process_idx == 0:
            self.logger.debug('pi_loss:%s Q_loss:%s',
                              pi_loss.data, Q_loss.data)

        return pi_loss + F.reshape(Q_loss, pi_loss.data.shape)

    def update(self, t_start, t_stop, R, states, actions, rewards, values,
               action_values, action_distribs, action_distribs_mu,
               avg_action_distribs):

        assert np.isscalar(R)

        total_loss = self.compute_loss(
            t_start=t_start,
            t_stop=t_stop,
            R=R,
            states=states,
            actions=actions,
            rewards=rewards,
            values=values,
            action_values=action_values,
            action_distribs=action_distribs,
            action_distribs_mu=action_distribs_mu,
            avg_action_distribs=avg_action_distribs)

        # Compute gradients using thread-specific model
        self.model.zerograds()
        total_loss.backward()
        # Copy the gradients to the globally shared model
        self.shared_model.zerograds()
        copy_param.copy_grad(
            target_link=self.shared_model, source_link=self.model)
        # Update the globally shared model
        if self.process_idx == 0:
            norm = sum(np.sum(np.square(param.grad))
                       for param in self.optimizer.target.params())
            self.logger.debug('grad norm:%s', norm)
        self.optimizer.update()

        self.sync_parameters()
        if isinstance(self.model, Recurrent):
            self.model.unchain_backward()

    def update_from_replay(self):

        if self.replay_buffer is None:
            return

        if len(self.replay_buffer) < self.replay_start_size:
            return

        episode = self.replay_buffer.sample_episodes(1, self.t_max)[0]

        with state_reset(self.model):
            with state_reset(self.shared_average_model):
                rewards = {}
                states = {}
                actions = {}
                action_distribs = {}
                action_distribs_mu = {}
                avg_action_distribs = {}
                action_values = {}
                values = {}
                for t, transition in enumerate(episode):
                    s = self.phi(transition['state'])
                    a = transition['action']
                    bs = np.expand_dims(s, 0)
                    action_distrib, action_value, v = self.model(bs)
                    with chainer.no_backprop_mode():
                        avg_action_distrib, _, _ = \
                            self.shared_average_model(bs)
                    states[t] = s
                    actions[t] = a
                    values[t] = v
                    action_distribs[t] = action_distrib
                    avg_action_distribs[t] = avg_action_distrib
                    rewards[t] = transition['reward']
                    action_distribs_mu[t] = transition['mu']
                    action_values[t] = action_value
                last_transition = episode[-1]
                if last_transition['is_state_terminal']:
                    R = 0
                else:
                    with chainer.no_backprop_mode():
                        last_s = last_transition['next_state']
                        action_distrib, action_value, last_v = self.model(
                            np.expand_dims(self.phi(last_s), 0))
                    R = float(last_v.data)
                return self.update(
                    R=R, t_start=0, t_stop=len(episode),
                    states=states, rewards=rewards,
                    actions=actions,
                    values=values,
                    action_distribs=action_distribs,
                    action_distribs_mu=action_distribs_mu,
                    avg_action_distribs=avg_action_distribs,
                    action_values=action_values)

    def update_on_policy(self, statevar):
        assert self.t_start < self.t

        if not self.disable_online_update:
            if statevar is None:
                R = 0
            else:
                with chainer.no_backprop_mode():
                    with state_kept(self.model):
                        action_distrib, action_value, v = self.model(statevar)
                R = float(v.data)
            self.update(
                t_start=self.t_start, t_stop=self.t, R=R,
                states=self.past_states,
                actions=self.past_actions,
                rewards=self.past_rewards,
                values=self.past_values,
                action_values=self.past_action_values,
                action_distribs=self.past_action_distrib,
                action_distribs_mu=None,
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
        action_distrib, action_value, v = self.model(statevar)
        self.past_action_values[self.t] = action_value
        action = action_distrib.sample().data[0]

        # Save values for a later update
        self.past_values[self.t] = v
        self.past_action_distrib[self.t] = action_distrib
        with chainer.no_backprop_mode():
            avg_action_distrib, _, _ = self.shared_average_model(
                statevar)
        self.past_avg_action_distrib[self.t] = avg_action_distrib

        self.past_actions[self.t] = action

        self.t += 1

        if self.process_idx == 0:
            self.logger.debug('t:%s r:%s a:%s action_distrib:%s',
                              self.t, reward, action, action_distrib)
        # Update stats
        self.average_value += (
            (1 - self.average_value_decay) *
            (float(v.data[0]) - self.average_value))
        self.average_entropy += (
            (1 - self.average_entropy_decay) *
            (float(action_distrib.entropy.data[0]) - self.average_entropy))

        if self.replay_buffer is not None and self.last_state is not None:
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
            action_distrib, _, _ = self.model(statevar)
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
        if self.replay_buffer is not None:
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
