import copy
from logging import getLogger
import os

import numpy as np
import chainer
from chainer import serializers
from chainer import functions as F

import copy_param

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
                 keep_loss_scale_same=False):

        # Globally shared model
        self.shared_model = model

        # Thread specific model
        self.model = copy.deepcopy(self.shared_model)

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

        self.t = 0
        self.t_start = 0
        self.past_action_log_prob = {}
        self.past_action_entropy = {}
        self.past_states = {}
        self.past_rewards = {}
        self.past_values = {}

    def sync_parameters(self):
        copy_param.copy_param(target_link=self.model,
                              source_link=self.shared_model)

    def act(self, state, reward, is_state_terminal):

        if self.clip_reward:
            reward = np.clip(reward, -1, 1)

        if not is_state_terminal:
            statevar = chainer.Variable(np.expand_dims(self.phi(state), 0))

        self.past_rewards[self.t - 1] = reward

        if (is_state_terminal and self.t_start < self.t) \
                or self.t - self.t_start == self.t_max:

            assert self.t_start < self.t

            if is_state_terminal:
                R = 0
            else:
                _, vout = self.model.pi_and_v(statevar, keep_same_state=True)
                R = float(vout.data)

            pi_loss = 0
            v_loss = 0
            for i in reversed(range(self.t_start, self.t)):
                R *= self.gamma
                R += self.past_rewards[i]
                v = self.past_values[i]
                if self.process_idx == 0:
                    logger.debug('s:%s v:%s R:%s',
                                 self.past_states[i].data.sum(), v.data, R)
                advantage = R - v
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

        if not is_state_terminal:
            self.past_states[self.t] = statevar
            pout, vout = self.model.pi_and_v(statevar)
            self.past_action_log_prob[self.t] = pout.sampled_actions_log_probs
            self.past_action_entropy[self.t] = pout.entropy
            self.past_values[self.t] = vout
            self.t += 1
            if self.process_idx == 0:
                logger.debug('t:%s entropy:%s, probs:%s',
                             self.t, pout.entropy.data, pout.probs.data)
            return pout.action_indices[0]
        else:
            self.model.reset_state()
            return None

    def load_model(self, model_filename):
        """Load a network model form a file
        """
        serializers.load_hdf5(model_filename, self.model)
        copy_param.copy_param(target_link=self.model,
                              source_link=self.shared_model)
        opt_filename = model_filename + '.opt'
        if os.path.exists(opt_filename):
            print('WARNING: {0} was not found, so loaded only a model'.format(
                opt_filename))
            serializers.load_hdf5(model_filename + '.opt', self.optimizer)

    def save_model(self, model_filename):
        """Save a network model to a file
        """
        serializers.save_hdf5(model_filename, self.model)
        serializers.save_hdf5(model_filename + '.opt', self.optimizer)
