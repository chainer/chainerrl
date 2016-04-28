import copy
from logging import getLogger
logger = getLogger(__name__)
import os

import numpy as np
import chainer
from chainer import functions as F
from chainer import serializers

import agent
import smooth_l1_loss
import copy_param
import clipped_loss


class A3C(agent.Agent):
    """A3C: Asynchronous Advantage Actor-Critic.

    See http://arxiv.org/abs/1602.01783
    """

    def __init__(self, model, optimizer, t_max, gamma, beta=1e-2,
                 process_idx=0, clip_delta=False, clip_reward=True,
                 phi=lambda x: x):

        assert len(model) == 2

        # Globally shared model
        self.shared_model = model
        self.shared_policy = self.shared_model[0]
        self.shared_v_function = self.shared_model[1]

        # Thread specific model
        self.model = copy.deepcopy(self.shared_model)
        self.policy = self.model[0]
        self.v_function = self.model[1]

        self.optimizer = optimizer
        self.t_max = t_max
        self.gamma = gamma
        self.beta = beta
        self.process_idx = process_idx
        self.clip_delta = clip_delta
        self.clip_reward = clip_reward
        self.phi = phi

        self.t = 0
        self.t_start = 0
        self.past_action_log_prob = {}
        self.past_action_entropy = {}
        self.past_states = {}
        self.past_rewards = {}

    def sync_parameters(self):
        copy_param.copy_param(self.model, self.shared_model)

    def act(self, state, reward, is_state_terminal):

        if self.clip_reward:
            reward = np.clip(reward, -1, 1)

        state = np.expand_dims(self.phi(state), 0)

        self.past_rewards[self.t - 1] = reward

        if (is_state_terminal and self.t_start < self.t) \
                or self.t - self.t_start == self.t_max:

            assert self.t_start < self.t

            if is_state_terminal:
                R = 0
            else:
                R = float(self.v_function(state).data)

            pi_loss = 0
            v_loss = 0
            for i in reversed(range(self.t_start, self.t)):
                R *= self.gamma
                R += self.past_rewards[i]
                v = self.v_function(self.past_states[i])
                if self.process_idx == 0:
                    logger.debug('s:%s v:%s R:%s', self.past_states[
                                 i].sum(), v.data, R)
                advantage = R - v
                # Accumulate gradients of policy
                log_prob = self.past_action_log_prob[i]
                entropy = self.past_action_entropy[i]

                # Log probability is increased proportionally to advantage
                pi_loss -= log_prob * float(advantage.data)
                # Entropy is maximized
                pi_loss -= self.beta * entropy
                # Accumulate gradients of value function

                # Squared loss is used in the original paper, but here I
                # try smooth L1 loss as in the Nature DQN paper.
                if self.clip_delta:
                    v_loss += clipped_loss.clipped_loss(
                        v,
                        chainer.Variable(np.asarray([[R]], dtype=np.float32)))
                else:
                    v_loss += (v - R) ** 2 / 2

            pi_loss *= 0.5

            # Do we need to normalize losses by (self.t - self.t_start)?
            # Otherwise, loss scales can be different in case of self.t_max
            # and in case of termination.

            # I'm not sure but if we need to normalize losses...
            pi_loss /= self.t - self.t_start
            v_loss /= self.t - self.t_start

            if self.process_idx == 0:
                logger.debug('pi_loss:%s v_loss:%s', pi_loss.data, v_loss.data)

            # Compute gradients using thread-specific model
            self.model.zerograds()
            pi_loss.backward()
            v_loss.backward()
            # Copy the gradients to the globally shared model
            self.shared_model.zerograds()
            copy_param.copy_grad(self.shared_model, self.model)
            # Update the globally shared model
            if self.process_idx == 0:
                norm = self.optimizer.compute_grads_norm()
                logger.debug('grad norm:%s', norm)
            self.optimizer.update()
            if self.process_idx == 0:
                logger.debug('update')

            self.sync_parameters()

            self.past_action_log_prob = {}
            self.past_action_entropy = {}
            self.past_states = {}
            self.past_rewards = {}

            self.t_start = self.t

        if not is_state_terminal:
            self.past_states[self.t] = state
            action, log_prob, entropy, probs = \
                self.policy.sample_with_log_probability_and_entropy(state)
            self.past_action_log_prob[self.t] = log_prob
            self.past_action_entropy[self.t] = entropy
            self.t += 1
            if self.process_idx == 0:
                logger.debug('t:%s entropy:%s, probs:%s',
                             self.t, entropy.data, probs.data)
            return action[0]
        else:
            return None

    @property
    def links(self):
        return [self.shared_model]

    @property
    def optimizers(self):
        return [self.optimizer]

    def load_model(self, model_filename):
        """Load a network model form a file
        """
        serializers.load_hdf5(model_filename, self.model)
        copy_param.copy_param(self.model, self.shared_model)
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
