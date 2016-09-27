from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import range
from future import standard_library
standard_library.install_aliases()
import copy
from logging import getLogger
import os

import numpy as np
import chainer
from chainer import serializers
from chainer import functions as F

import copy_param
import async

logger = getLogger(__name__)


class A3CModel(chainer.Link):

    def pi_and_v(self, state, keep_same_state=False):
        raise NotImplementedError()

    def reset_state(self):
        pass

    def unchain_backward(self):
        pass


class A3C(object):
    """A3C: Asynchronous Advantage Actor-Critic.

    See http://arxiv.org/abs/1602.01783
    """

    def __init__(self, model, optimizer, t_max, gamma, beta=1e-2,
                 process_idx=0, clip_reward=True, phi=lambda x: x,
                 pi_loss_coef=1.0, v_loss_coef=0.5,
                 keep_loss_scale_same=False, use_terminal_state_value=False,
                 normalize_grad_by_t_max=False,
                 use_average_reward=False, average_reward_tau=1e-2):

        assert isinstance(model, A3CModel)
        # Globally shared model
        self.shared_model = model

        # Thread specific model
        self.model = copy.deepcopy(self.shared_model)
        async.assert_params_not_shared(self.shared_model, self.model)

        self.optimizer = optimizer

        self.t_max = t_max
        self.gamma = gamma
        self.beta = beta
        self.process_idx = process_idx
        self.clip_reward = clip_reward
        self.phi = phi
        self.pi_loss_coef = pi_loss_coef
        self.v_loss_coef = v_loss_coef
        self.keep_loss_scale_same = keep_loss_scale_same
        self.use_terminal_state_value = use_terminal_state_value
        self.normalize_grad_by_t_max = normalize_grad_by_t_max
        self.use_average_reward = use_average_reward
        self.average_reward_tau = average_reward_tau

        self.t = 0
        self.t_start = 0
        self.past_action_log_prob = {}
        self.past_action_entropy = {}
        self.past_states = {}
        self.past_rewards = {}
        self.past_values = {}
        self.average_reward = 0
        # A3C won't use a explorer, but this arrtibute is referenced by run_dqn
        self.explorer = None

    def sync_parameters(self):
        copy_param.copy_param(target_link=self.model,
                              source_link=self.shared_model)

    def update(self, statevar):
        assert self.t_start < self.t

        if statevar is None:
            R = 0
        else:
            _, vout = self.model.pi_and_v(statevar, keep_same_state=True)
            R = float(vout.data)

        pi_loss = 0
        v_loss = 0
        for i in reversed(range(self.t_start, self.t)):
            R *= self.gamma
            R += self.past_rewards[i]
            if self.use_average_reward:
                R -= self.average_reward
            v = self.past_values[i]
            if self.process_idx == 0:
                logger.debug('t:%s s:%s v:%s R:%s',
                             i, self.past_states[i].data.sum(), v.data, R)
            advantage = R - v
            if self.use_average_reward:
                self.average_reward += self.average_reward_tau * float(advantage.data)
            # Accumulate gradients of policy
            log_prob = self.past_action_log_prob[i]
            entropy = self.past_action_entropy[i]

            # Log probability is increased proportionally to advantage
            pi_loss -= log_prob * float(advantage.data)
            # Entropy is maximized
            pi_loss -= self.beta * entropy
            # Accumulate gradients of value function

            v_loss += (v - R) ** 2 / 2

        if self.pi_loss_coef != 1.0:
            pi_loss *= self.pi_loss_coef

        if self.v_loss_coef != 1.0:
            v_loss *= self.v_loss_coef

        # Normalize the loss of sequences truncated by terminal states
        if self.keep_loss_scale_same and \
                self.t - self.t_start < self.t_max:
            factor = self.t_max / (self.t - self.t_start)
            pi_loss *= factor
            v_loss *= factor

        if self.normalize_grad_by_t_max:
            pi_loss /= self.t - self.t_start
            v_loss /= self.t - self.t_start

        if self.process_idx == 0:
            logger.debug('pi_loss:%s v_loss:%s', pi_loss.data, v_loss.data)

        total_loss = pi_loss + F.reshape(v_loss, pi_loss.data.shape)

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
        if self.process_idx == 0:
            logger.debug('update')

        self.sync_parameters()
        self.model.unchain_backward()

        self.past_action_log_prob = {}
        self.past_action_entropy = {}
        self.past_states = {}
        self.past_rewards = {}
        self.past_values = {}

        self.t_start = self.t


    def act(self, state, reward):

        if self.clip_reward:
            reward = np.clip(reward, -1, 1)

        statevar = chainer.Variable(np.expand_dims(self.phi(state), 0))

        self.past_rewards[self.t - 1] = reward

        if self.t - self.t_start == self.t_max:
            self.update(statevar)

        self.past_states[self.t] = statevar
        pout, vout = self.model.pi_and_v(statevar)
        self.past_action_log_prob[self.t] = pout.sampled_actions_log_probs
        self.past_action_entropy[self.t] = pout.entropy
        self.past_values[self.t] = vout
        self.t += 1
        action = pout.sampled_actions.data[0]
        if self.process_idx == 0:
            logger.debug('t:%s r:%s a:%s pout:%s', self.t, reward, action, pout)
        return action

    def observe_terminal(self, state, reward):
        if self.clip_reward:
            reward = np.clip(reward, -1, 1)

        self.past_rewards[self.t - 1] = reward
        self.update(None)

        self.model.reset_state()

    def stop_current_episode(self, state, reward):
        if self.clip_reward:
            reward = np.clip(reward, -1, 1)

        self.past_rewards[self.t - 1] = reward
        statevar = chainer.Variable(np.expand_dims(self.phi(state), 0))
        self.update(statevar)

        self.model.reset_state()

    def load_model(self, model_filename):
        """Load a network model form a file
        """
        serializers.load_hdf5(model_filename, self.model)
        copy_param.copy_param(target_link=self.shared_model,
                              source_link=self.model)
        opt_filename = model_filename + '.opt'
        if os.path.exists(opt_filename):
            serializers.load_hdf5(model_filename + '.opt', self.optimizer)
        else:
            print('WARNING: {0} was not found, so loaded only a model'.format(
                opt_filename))

    def save_model(self, model_filename):
        """Save a network model to a file
        """
        serializers.save_hdf5(model_filename, self.model)
        serializers.save_hdf5(model_filename + '.opt', self.optimizer)

    def get_stats_keys(self):
        return ()

    def get_stats_values(self):
        return ()
