from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
from builtins import super
standard_library.install_aliases()

import os

import numpy as np
import chainer
import chainer.functions as F
from chainer import cuda
from chainer import serializers

from . import dqn


class PGT(dqn.DQN):
    """Policy Gradient Theorem.

    This algorithm optimizes a Q-function and a stochastic policy based on
    policy gradients computed by the policy gradient theorem. Unlike DDPG and
    SVG(0), it does not use value grdients.
    """

    def __init__(self, model, actor_optimizer, critic_optimizer, replay_buffer,
                 gamma, explorer, beta=1e-2, **kwargs):
        super().__init__(model, None, replay_buffer, gamma, explorer, **kwargs)

        # Aliases for convenience
        self.q_function = self.model['q_function']
        self.policy = self.model['policy']
        self.target_q_function = self.target_model['q_function']
        self.target_policy = self.target_model['policy']

        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.beta = beta

    def update(self, experiences, errors_out=None):
        """Update the model from experiences
        """

        batch_size = len(experiences)

        # Store necessary data in arrays
        batch_state = self._batch_states(
            [elem['state'] for elem in experiences])

        batch_actions = self.xp.asarray(
            [elem['action'] for elem in experiences])

        batch_next_state = self._batch_states(
            [elem['next_state'] for elem in experiences])

        batch_rewards = self.xp.asarray(
            [[elem['reward']] for elem in experiences], dtype=np.float32)

        batch_terminal = self.xp.asarray(
            [[elem['is_state_terminal']] for elem in experiences],
            dtype=np.float32)

        # Update Q-function
        def compute_critic_loss():
            next_actions = self.target_policy(
                batch_next_state, test=True).sampled_actions
            next_q = self.target_q_function(batch_next_state, next_actions,
                                            test=True)

            target_q = batch_rewards + self.gamma * \
                (1.0 - batch_terminal) * next_q
            target_q.creator = None

            predict_q = self.q_function(batch_state, batch_actions, test=False)

            return F.mean_squared_error(target_q, predict_q)

        def compute_actor_loss():
            pout = self.policy(batch_state, test=False)
            q = self.q_function(batch_state, pout.sampled_actions, test=True)
            log_probs = pout.sampled_actions_log_probs
            v = self.q_function(
                batch_state, pout.most_probable_actions, test=True)
            advantage = F.reshape(q - v, (batch_size,))
            advantage = chainer.Variable(advantage.data)
            return - F.sum(advantage * log_probs + self.beta * pout.entropy) \
                / batch_size

        self.critic_optimizer.update(compute_critic_loss)
        self.actor_optimizer.update(compute_actor_loss)

    def select_greedy_action(self, state):

        s = self._batch_states([state])
        action = self.policy(s, test=True).sampled_actions
        # Q is not needed here, but log it just for information
        q = self.q_function(s, action, test=True)
        self.logger.debug('t:%s a:%s q:%s',
                          self.t, action.data[0], q.data)
        return cuda.to_cpu(action.data[0])

    def select_action(self, state):
        return self.explorer.select_action(
            self.t, lambda: self.select_greedy_action(state))

    def save_model(self, model_filename):
        """Save a network model to a file."""

        serializers.save_hdf5(model_filename, self.model)
        serializers.save_hdf5(model_filename + '.opt.actor',
                              self.actor_optimizer)
        serializers.save_hdf5(model_filename + '.opt.critic',
                              self.critic_optimizer)

    def load_model(self, model_filename):
        """Load a network model form a file."""

        serializers.load_hdf5(model_filename, self.model)
        self.sync_target_network()

        actor_opt_filename = model_filename + '.opt.actor'
        if os.path.exists(actor_opt_filename):
            print('WARNING: {0} was not found'.format(actor_opt_filename))
            serializers.load_hdf5(actor_opt_filename, self.actor_optimizer)

        critic_opt_filename = model_filename + '.opt.critic'
        if os.path.exists(critic_opt_filename):
            print('WARNING: {0} was not found'.format(critic_opt_filename))
            serializers.load_hdf5(critic_opt_filename, self.critic_optimizer)
