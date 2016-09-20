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
import chainer.links as L
import chainer.functions as F
from chainer import cuda
from chainer import serializers

from . import dqn
import copy_param


class DDPGModel(chainer.Chain):

    def __init__(self, policy, q_func):
        super().__init__(policy=policy, q_function=q_func)

    def push_state(self):
        self.policy.push_state()
        self.q_function.push_state()

    def pop_state(self):
        self.policy.pop_state()
        self.q_function.pop_state()

    def reset_state(self):
        self.policy.reset_state()
        self.q_function.reset_state()


class DDPG(dqn.DQN):
    """Deep Deterministic Policy Gradients.
    """

    def __init__(self, model, actor_optimizer, critic_optimizer, replay_buffer,
                 gamma, explorer, **kwargs):
        super().__init__(model, None, replay_buffer, gamma, explorer, **kwargs)

        # Aliases for convenience
        self.q_function = self.model['q_function']
        self.policy = self.model['policy']
        self.target_q_function = self.target_model['q_function']
        self.target_policy = self.target_model['policy']

        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

        self.average_actor_loss = 0.0
        self.average_critic_loss = 0.0

    # Update Q-function
    def compute_critic_loss(self, batch):
        """
        target_q_function must have seen s_t and a_t
        target_policy must have seen s_t
        q_function must have seen s_{t-1}
        """

        batch_next_state = batch['next_state']
        batch_rewards = batch['reward']
        batch_terminal = batch['is_state_terminal']
        batch_state = batch['state']
        batch_actions = batch['action']
        batch_next_actions = batch['next_action']

        # Target policy observes s_{t+1}
        next_actions = self.target_policy(batch_next_state, test=True)

        self.target_q_function.push_and_keep_state()
        next_q = self.target_q_function(batch_next_state, next_actions,
                                        test=True)
        self.target_q_function.pop_state()

        # Target Q-function observes s_{t+1} and a_{t+1}
        self.target_q_function.update_state(
            batch_next_state, batch_next_actions, test=True)

        target_q = batch_rewards + self.gamma * \
            (1.0 - batch_terminal) * next_q
        target_q.creator = None

        # Estimated Q-function observes s_t and a_t
        predict_q = self.q_function(batch_state, batch_actions, test=False)

        loss = F.mean_squared_error(target_q, predict_q)

        # Update stats
        self.average_critic_loss *= self.average_loss_decay
        self.average_critic_loss += ((1 - self.average_loss_decay) *
                                     float(loss.data))

        return loss

    def compute_actor_loss(self, batch):
        """
        q_function must have seen s_{t-1}
        policy must have seen s_{t-1}
        """

        batch_state = batch['state']
        batch_size = batch_state.shape[0]

        # Estimated policy observes s_t
        onpolicy_actions = self.policy(batch_state, test=False)

        # Estimated Q-function must have not seen s_t so far
        # self.q_function.push_and_keep_state()
        q = self.q_function(batch_state, onpolicy_actions, test=True)
        # self.q_function.pop_state()

        # Since we want to maximize Q, loss is negation of Q
        loss = - F.sum(q) / batch_size

        # Update stats
        self.average_actor_loss *= self.average_loss_decay
        self.average_actor_loss += ((1 - self.average_loss_decay) *
                                    float(loss.data))
        return loss

    def update(self, experiences, errors_out=None):
        """Update the model from experiences
        """

        batch = dqn.batch_experiences(experiences, self.xp, self.phi)
        self.critic_optimizer.update(lambda: self.compute_critic_loss(batch))
        self.actor_optimizer.update(lambda: self.compute_actor_loss(batch))

    def update_from_episodes(self, episodes, errors_out=None):
        self.model.push_state()
        self.target_model.push_state()

        sorted_episodes = list(reversed(sorted(episodes, key=len)))
        max_epi_len = len(sorted_episodes[0])

        # actor_loss = 0
        # critic_loss = 0
        # for i in range(max_epi_len):
        #     transitions = []
        #     for ep in sorted_episodes:
        #         if len(ep) <= i:
        #             break
        #         transitions.append(ep[i])
        #     batch = dqn.batch_experiences(transitions, xp=self.xp, phi=self.phi)
        #     if i == 0:
        #         self.input_initial_batch_target_model(batch)
        #
        #     actor_loss += self.compute_actor_loss(batch)
        #     critic_loss += self.compute_critic_loss(batch)
        #
        # actor_loss /= max_epi_len
        # critic_loss /= max_epi_len
        # self.actor_optimizer.zero_grads()
        # actor_loss.backward()
        # self.critic_optimizer.zero_grads()
        # critic_loss.backward()
        # # self.actor_optimizer.update(lambda: actor_loss / max_epi_len)
        # # self.critic_optimizer.update(lambda: critic_loss / max_epi_len)
        # self.actor_optimizer.update()
        # self.critic_optimizer.update()

        # Precompute all the input batches
        batches = []
        for i in range(max_epi_len):
            transitions = []
            for ep in sorted_episodes:
                if len(ep) <= i:
                    break
                transitions.append(ep[i])
            batch = dqn.batch_experiences(transitions, xp=self.xp, phi=self.phi)
            batches.append(batch)

        self.input_initial_batch_target_model(batches[0])

        # Update critic through time
        critic_loss = 0
        for batch in batches:
            critic_loss += self.compute_critic_loss(batch)
        self.critic_optimizer.update(lambda: critic_loss / max_epi_len)

        self.model.reset_state()
        self.target_model.reset_state()

        # Update actor through time
        actor_loss = 0
        for batch in batches:
            actor_loss += self.compute_actor_loss(batch)
        self.actor_optimizer.update(lambda: actor_loss / max_epi_len)

        # actor_loss /= max_epi_len
        # self.actor_optimizer.zero_grads()
        # actor_loss.backward()
        # self.critic_optimizer.zero_grads()
        # critic_loss.backward()
        # # self.actor_optimizer.update(lambda: actor_loss / max_epi_len)
        # self.actor_optimizer.update()
        # self.critic_optimizer.update()

        self.model.pop_state()
        self.target_model.pop_state()

    def select_greedy_action(self, state):

        s = self._batch_states([state])
        action = self.policy(s, test=True)
        # Q is not needed here, but log it just for information
        q = self.q_function(s, action, test=True)

        # Update stats
        self.average_q *= self.average_q_decay
        self.average_q += (1 - self.average_q_decay) * float(q.data)

        self.logger.debug('t:%s a:%s q:%s',
                          self.t, action.data[0], q.data)
        return cuda.to_cpu(action.data[0])

    def select_action(self, state):
        return self.explorer.select_action(
            self.t, lambda: self.select_greedy_action(state))

    def save_model(self, model_filename):
        """Save a network model to a file."""

        serializers.save_hdf5(model_filename, self.model)
        serializers.save_hdf5(model_filename + '.target', self.target_model)
        serializers.save_hdf5(model_filename + '.opt.actor',
                              self.actor_optimizer)
        serializers.save_hdf5(model_filename + '.opt.critic',
                              self.critic_optimizer)

    def load_model(self, model_filename):
        """Load a network model form a file."""

        serializers.load_hdf5(model_filename, self.model)

        # Load target model
        target_filename = model_filename + '.target'
        if os.path.exists(target_filename):
            serializers.load_hdf5(target_filename, self.target_model)
        else:
            print('WARNING: {0} was not found'.format(target_filename))
            copy_param.copy_param(target_link=self.target_model,
                                  source_link=self.model)

        actor_opt_filename = model_filename + '.opt.actor'
        if os.path.exists(actor_opt_filename):
            serializers.load_hdf5(actor_opt_filename, self.actor_optimizer)
        else:
            print('WARNING: {0} was not found'.format(actor_opt_filename))

        critic_opt_filename = model_filename + '.opt.critic'
        if os.path.exists(critic_opt_filename):
            serializers.load_hdf5(critic_opt_filename, self.critic_optimizer)
        else:
            print('WARNING: {0} was not found'.format(critic_opt_filename))

    def get_stats_keys(self):
        return ('average_q', 'average_actor_loss', 'average_critic_loss')

    def get_stats_values(self):
        return (self.average_q, self.average_actor_loss, self.average_critic_loss)

    def input_initial_batch_target_model(self, batch):
        self.target_q_function.update_state(batch['state'], batch['action'], test=True)
        self.target_policy(batch['state'])
