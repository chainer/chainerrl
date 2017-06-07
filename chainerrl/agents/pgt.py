from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
from builtins import *  # NOQA
standard_library.install_aliases()

import copy
from logging import getLogger

import chainer
from chainer import cuda
import chainer.functions as F
import numpy as np

from chainerrl.agent import Agent
from chainerrl.agent import AttributeSavingMixin
from chainerrl.agents.ddpg import disable_train
from chainerrl.misc.batch_states import batch_states
from chainerrl.misc.copy_param import synchronize_parameters
from chainerrl.recurrent import Recurrent
from chainerrl.replay_buffer import ReplayUpdater


class PGT(AttributeSavingMixin, Agent):
    """Policy Gradient Theorem with an approximate policy and a Q-function.

    This agent is almost the same with DDPG except that it uses the likelihood
    ratio gradient estimation instead of value gradients.

    Args:
        model (chainer.Chain): Chain that contains both a policy and a
            Q-function
        actor_optimizer (Optimizer): Optimizer setup with the policy
        critic_optimizer (Optimizer): Optimizer setup with the Q-function
        replay_buffer (ReplayBuffer): Replay buffer
        gamma (float): Discount factor
        explorer (Explorer): Explorer that specifies an exploration strategy.
        gpu (int): GPU device id. -1 for CPU.
        replay_start_size (int): if the replay buffer's size is less than
            replay_start_size, skip update
        minibatch_size (int): Minibatch size
        update_interval (int): Model update interval in step
        target_update_interval (int): Target model update interval in step
        phi (callable): Feature extractor applied to observations
        target_update_method (str): 'hard' or 'soft'.
        soft_update_tau (float): Tau of soft target update.
        n_times_update (int): Number of repetition of update
        average_q_decay (float): Decay rate of average Q, only used for
            recording statistics
        average_loss_decay (float): Decay rate of average loss, only used for
            recording statistics
        batch_accumulator (str): 'mean' or 'sum'
        logger (Logger): Logger used
        beta (float): Coefficient for entropy regularization
        act_deterministically (bool): Act deterministically by selecting most
            probable actions in test time
        batch_states (callable): method which makes a batch of observations.
            default is `chainerrl.misc.batch_states.batch_states`
    """

    saved_attributes = ('model',
                        'target_model',
                        'actor_optimizer',
                        'critic_optimizer')

    def __init__(self, model, actor_optimizer, critic_optimizer, replay_buffer,
                 gamma, explorer, beta=1e-2, act_deterministically=False,
                 gpu=-1, replay_start_size=50000,
                 minibatch_size=32, update_interval=1,
                 target_update_interval=10000,
                 phi=lambda x: x,
                 target_update_method='hard',
                 soft_update_tau=1e-2,
                 n_times_update=1, average_q_decay=0.999,
                 average_loss_decay=0.99,
                 logger=getLogger(__name__),
                 batch_states=batch_states):

        self.model = model

        if gpu is not None and gpu >= 0:
            cuda.get_device(gpu).use()
            self.model.to_gpu(device=gpu)

        self.xp = self.model.xp
        self.replay_buffer = replay_buffer
        self.gamma = gamma
        self.explorer = explorer
        self.gpu = gpu
        self.target_update_interval = target_update_interval
        self.phi = phi
        self.target_update_method = target_update_method
        self.soft_update_tau = soft_update_tau
        self.logger = logger
        self.average_q_decay = average_q_decay
        self.average_loss_decay = average_loss_decay
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.beta = beta
        self.act_deterministically = act_deterministically
        self.replay_updater = ReplayUpdater(
            replay_buffer=replay_buffer,
            update_func=self.update,
            batchsize=minibatch_size,
            episodic_update=False,
            n_times_update=n_times_update,
            replay_start_size=replay_start_size,
            update_interval=update_interval,
        )
        self.batch_states = batch_states

        self.t = 0
        self.last_state = None
        self.last_action = None
        self.target_model = copy.deepcopy(self.model)
        disable_train(self.target_model['q_function'])
        disable_train(self.target_model['policy'])
        self.average_q = 0
        self.average_actor_loss = 0.0
        self.average_critic_loss = 0.0

        # Aliases for convenience
        self.q_function = self.model['q_function']
        self.policy = self.model['policy']
        self.target_q_function = self.target_model['q_function']
        self.target_policy = self.target_model['policy']

        self.sync_target_network()

    def sync_target_network(self):
        """Synchronize target network with current network."""
        synchronize_parameters(
            src=self.model,
            dst=self.target_model,
            method=self.target_update_method,
            tau=self.soft_update_tau)

    def update(self, experiences, errors_out=None):
        """Update the model from experiences."""

        batch_size = len(experiences)

        # Store necessary data in arrays
        batch_state = self.batch_states(
            [elem['state'] for elem in experiences], self.xp, self.phi)

        batch_actions = self.xp.asarray(
            [elem['action'] for elem in experiences])

        batch_next_state = self.batch_states(
            [elem['next_state'] for elem in experiences], self.xp, self.phi)

        batch_rewards = self.xp.asarray(
            [[elem['reward']] for elem in experiences], dtype=np.float32)

        batch_terminal = self.xp.asarray(
            [[elem['is_state_terminal']] for elem in experiences],
            dtype=np.float32)

        # Update Q-function
        def compute_critic_loss():

            with chainer.no_backprop_mode():
                pout = self.target_policy(batch_next_state)
                next_actions = pout.sample()
                next_q = self.target_q_function(batch_next_state, next_actions)

                target_q = batch_rewards + self.gamma * \
                    (1.0 - batch_terminal) * next_q

            predict_q = self.q_function(batch_state, batch_actions)

            loss = F.mean_squared_error(target_q, predict_q)

            # Update stats
            self.average_critic_loss *= self.average_loss_decay
            self.average_critic_loss += ((1 - self.average_loss_decay) *
                                         float(loss.data))

            return loss

        def compute_actor_loss():
            pout = self.policy(batch_state)
            sampled_actions = pout.sample().data
            log_probs = pout.log_prob(sampled_actions)
            with chainer.using_config('train', False):
                q = self.q_function(batch_state, sampled_actions)
                v = self.q_function(
                    batch_state, pout.most_probable)
            advantage = F.reshape(q - v, (batch_size,))
            advantage = chainer.Variable(advantage.data)
            loss = - F.sum(advantage * log_probs + self.beta * pout.entropy) \
                / batch_size

            # Update stats
            self.average_actor_loss *= self.average_loss_decay
            self.average_actor_loss += ((1 - self.average_loss_decay) *
                                        float(loss.data))

            return loss

        self.critic_optimizer.update(compute_critic_loss)
        self.actor_optimizer.update(compute_actor_loss)

    def act_and_train(self, state, reward):

        self.logger.debug('t:%s r:%s', self.t, reward)

        greedy_action = self.act(state)
        action = self.explorer.select_action(self.t, lambda: greedy_action)
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

        return self.last_action

    def act(self, state):

        with chainer.using_config('train', False):
            s = self.batch_states([state], self.xp, self.phi)
            if self.act_deterministically:
                action = self.policy(s).most_probable
            else:
                action = self.policy(s).sample()
            # Q is not needed here, but log it just for information
            q = self.q_function(s, action)

        # Update stats
        self.average_q *= self.average_q_decay
        self.average_q += (1 - self.average_q_decay) * float(q.data)

        self.logger.debug('t:%s a:%s q:%s',
                          self.t, action.data[0], q.data)
        return cuda.to_cpu(action.data[0])

    def stop_episode_and_train(self, state, reward, done=False):

        assert self.last_state is not None
        assert self.last_action is not None

        # Add a transition to the replay buffer
        self.replay_buffer.append(
            state=self.last_state,
            action=self.last_action,
            reward=reward,
            next_state=state,
            next_action=self.last_action,
            is_state_terminal=done)

        self.stop_episode()

    def stop_episode(self):
        self.last_state = None
        self.last_action = None
        if isinstance(self.model, Recurrent):
            self.model.reset_state()
        self.replay_buffer.stop_current_episode()

    def select_action(self, state):
        return self.explorer.select_action(
            self.t, lambda: self.act(state))

    def get_statistics(self):
        return [
            ('average_q', self.average_q),
            ('average_actor_loss', self.average_actor_loss),
            ('average_critic_loss', self.average_critic_loss),
        ]
