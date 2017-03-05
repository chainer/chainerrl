from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

import copy
from logging import getLogger

import chainer
from chainer import functions as F
import numpy as np

from chainerrl import agent
from chainerrl.misc import async
from chainerrl.misc import copy_param
from chainerrl.recurrent import Recurrent
from chainerrl.recurrent import state_kept
from chainerrl.recurrent import state_reset


def asfloat(x):
    if isinstance(x, chainer.Variable):
        return float(x.data)
    else:
        return float(x)


class PCL(agent.AttributeSavingMixin, agent.AsyncAgent):
    """PCL (Path Consistency Learning).

    See https://arxiv.org/abs/1702.08892

    Args:
        model (ACERModel): Model to train. It must be a callable that accepts
            observations as input and return three values: action distributions
            (Distribution), Q values (ActionValue) and state values
            (chainer.Variable).
        optimizer (chainer.Optimizer): optimizer used to train the model
        t_max (int): The model is updated after every t_max local steps
        gamma (float): Discount factor [0,1]
        beta (float): Weight coefficient for the entropy regularizaiton term.
        phi (callable): Feature extractor function
        pi_loss_coef (float): Weight coefficient for the loss of the policy
        v_loss_coef (float): Weight coefficient for the loss of the value
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

    def __init__(self, model, optimizer,
                 replay_buffer=None,
                 t_max=None,
                 gamma=0.99,
                 tau=1e-2,
                 process_idx=0, phi=lambda x: x,
                 pi_loss_coef=1.0, v_loss_coef=0.5,
                 rollout_len=10,
                 disable_online_update=False,
                 n_times_replay=1,
                 replay_start_size=10 ** 2,
                 normalize_loss_by_steps=True,
                 act_deterministically=False,
                 average_loss_decay=0.999,
                 average_entropy_decay=0.999,
                 average_value_decay=0.999,
                 explorer=None,
                 logger=None):

        # Globally shared model
        self.shared_model = model

        # Thread specific model
        self.model = copy.deepcopy(self.shared_model)
        async.assert_params_not_shared(self.shared_model, self.model)

        self.optimizer = optimizer

        self.replay_buffer = replay_buffer
        self.t_max = t_max
        self.gamma = gamma
        self.tau = tau
        self.phi = phi
        self.pi_loss_coef = pi_loss_coef
        self.v_loss_coef = v_loss_coef
        self.rollout_len = rollout_len
        self.normalize_loss_by_steps = normalize_loss_by_steps
        self.act_deterministically = act_deterministically
        self.disable_online_update = disable_online_update
        self.n_times_replay = n_times_replay
        self.replay_start_size = replay_start_size
        self.average_loss_decay = average_loss_decay
        self.average_value_decay = average_value_decay
        self.average_entropy_decay = average_entropy_decay
        self.logger = logger if logger else getLogger(__name__)

        self.t = 0
        self.last_state = None
        self.last_action = None
        self.explorer = explorer

        # Stats
        self.average_loss = 0
        self.average_value = 0
        self.average_entropy = 0

        self.init_history_data_for_online_update()

    def init_history_data_for_online_update(self):
        self.past_actions = {}
        self.past_rewards = {}
        self.past_values = {}
        self.past_action_distrib = {}
        self.t_start = self.t

    def sync_parameters(self):
        copy_param.copy_param(target_link=self.model,
                              source_link=self.shared_model)

    @property
    def shared_attributes(self):
        return ('shared_model', 'optimizer')

    def compute_loss(self, t_start, t_stop, R, actions, rewards, values,
                     action_distribs):

        seq_len = t_stop - t_start
        assert len(actions) == seq_len
        assert len(rewards) == seq_len
        assert len(values) == seq_len
        assert len(action_distribs) == seq_len

        assert np.isscalar(R)
        pi_loss = 0
        v_loss = 0
        for t in range(t_start, t_stop):
            d = min(t_stop - t, self.rollout_len)
            R_seq = sum(self.gamma ** i * rewards[t + i] for i in range(d))
            G = sum(self.gamma ** i * action_distribs[t + i].log_prob(
                np.expand_dims(actions[t + i], 0))
                for i in range(d))
            # log_probs = [self.gamma ** i * action_distribs[t + i].log_prob(
            #     np.expand_dims(actions[t + i], 0))
            #     for i in range(d)]
            # for i in range(1, d):
            #     log_probs[i] = log_probs[i].data
            # G = sum(log_probs)
            last_v = values[t + d] if t + d < t_stop else R
            # C_pi only backprop through pi
            C_pi = (- asfloat(values[t]) +
                    self.gamma ** d * asfloat(last_v) +
                    R_seq -
                    self.tau * G)
            # C_v only backprop through v
            C_v = (- values[t] +
                   self.gamma ** d * last_v +
                   # self.gamma ** d * asfloat(last_v) +
                   R_seq -
                   self.tau * asfloat(G))

            pi_loss += C_pi ** 2 / 2
            v_loss += C_v ** 2 / 2

        pi_loss *= self.pi_loss_coef
        v_loss *= self.v_loss_coef

        if self.normalize_loss_by_steps:
            pi_loss /= t_stop - t_start
            v_loss /= t_stop - t_start

        if self.process_idx == 0:
            self.logger.debug('pi_loss:%s v_loss:%s',
                              pi_loss.data, v_loss.data)

        return pi_loss + F.reshape(v_loss, pi_loss.data.shape)

    def update(self, t_start, t_stop, R, actions, rewards, values,
               action_distribs):

        assert np.isscalar(R)

        total_loss = self.compute_loss(
            t_start=t_start,
            t_stop=t_stop,
            R=R,
            actions=actions,
            rewards=rewards,
            values=values,
            action_distribs=action_distribs)

        self.average_loss += (
            (1 - self.average_loss_decay) *
            (float(total_loss.data[0]) - self.average_loss))

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

        episode = self.replay_buffer.sample_episodes(1, max_len=self.t_max)[0]

        with state_reset(self.model):
            rewards = {}
            actions = {}
            action_distribs = {}
            values = {}
            for t, transition in enumerate(episode):
                s = self.phi(transition['state'])
                a = transition['action']
                bs = np.expand_dims(s, 0)
                action_distrib, v = self.model(bs)
                actions[t] = a
                values[t] = v
                action_distribs[t] = action_distrib
                rewards[t] = transition['reward']
            last_transition = episode[-1]
            if last_transition['is_state_terminal']:
                R = 0
            else:
                with chainer.no_backprop_mode():
                    last_s = last_transition['next_state']
                    action_distrib, last_v = self.model(
                        np.expand_dims(self.phi(last_s), 0))
                R = float(last_v.data)
            return self.update(
                R=R,
                t_start=0,
                t_stop=len(episode),
                rewards=rewards,
                actions=actions,
                values=values,
                action_distribs=action_distribs)

    def update_on_policy(self, statevar):
        assert self.t_start < self.t

        if not self.disable_online_update:
            if statevar is None:
                R = 0
            else:
                with chainer.no_backprop_mode():
                    with state_kept(self.model):
                        action_distrib, v = self.model(statevar)
                R = float(v.data)
            self.update(
                t_start=self.t_start, t_stop=self.t, R=R,
                actions=self.past_actions,
                rewards=self.past_rewards,
                values=self.past_values,
                action_distribs=self.past_action_distrib)

        self.init_history_data_for_online_update()

    def act_and_train(self, state, reward):

        statevar = np.expand_dims(self.phi(state), 0)

        if self.last_state is not None:
            self.past_rewards[self.t - 1] = reward

        if self.t - self.t_start == self.t_max:
            self.update_on_policy(statevar)
            for _ in range(self.n_times_replay):
                self.update_from_replay()

        action_distrib, v = self.model(statevar)
        action = action_distrib.sample().data[0]
        if self.explorer is not None:
            action = self.explorer.select_action(self.t, lambda: action)

        # Save values for a later update
        self.past_values[self.t] = v
        self.past_actions[self.t] = action
        self.past_action_distrib[self.t] = action_distrib

        self.t += 1

        if self.process_idx == 0:
            self.logger.debug(
                't:%s r:%s a:%s action_distrib:%s v:%s',
                self.t, reward, action, action_distrib, float(v.data))
        # Update stats
        self.average_value += (
            (1 - self.average_value_decay) *
            (float(v.data[0]) - self.average_value))
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

    def load(self, dirname):
        super().load(dirname)
        copy_param.copy_param(target_link=self.shared_model,
                              source_link=self.model)

    def get_statistics(self):
        return [
            ('average_loss', self.average_loss),
            ('average_value', self.average_value),
            ('average_entropy', self.average_entropy),
        ]
