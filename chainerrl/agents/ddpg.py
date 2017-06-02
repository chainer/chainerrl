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

from chainerrl.agent import Agent
from chainerrl.agent import AttributeSavingMixin
from chainerrl.misc.batch_states import batch_states
from chainerrl.misc.copy_param import synchronize_parameters
from chainerrl.recurrent import Recurrent
from chainerrl.recurrent import RecurrentChainMixin
from chainerrl.recurrent import state_kept
from chainerrl.replay_buffer import batch_experiences
from chainerrl.replay_buffer import ReplayUpdater


def disable_train(chain):
    call_orig = chain.__call__

    def call_test(self, x):
        with chainer.using_config('train', False):
            return call_orig(self, x)

    chain.__call__ = call_test


class DDPGModel(chainer.Chain, RecurrentChainMixin):

    def __init__(self, policy, q_func):
        super().__init__(policy=policy, q_function=q_func)


class DDPG(AttributeSavingMixin, Agent):
    """Deep Deterministic Policy Gradients.

    This can be used as SVG(0) by specifying a Gaussina policy instead of a
    deterministic policy.

    Args:
        model (DDPGModel): DDPG model that contains both a policy and a
            Q-function
        actor_optimizer (Optimizer): Optimizer setup with the policy
        critic_optimizer (Optimizer): Optimizer setup with the Q-function
        replay_buffer (ReplayBuffer): Replay buffer
        gamma (float): Discount factor
        explorer (Explorer): Explorer that specifies an exploration strategy.
        gpu (int): GPU device id if not None nor negative.
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
        episodic_update (bool): Use full episodes for update if set True
        episodic_update_len (int or None): Subsequences of this length are used
            for update if set int and episodic_update=True
        logger (Logger): Logger used
        batch_states (callable): method which makes a batch of observations.
            default is `chainerrl.misc.batch_states.batch_states`
    """

    saved_attributes = ('model',
                        'target_model',
                        'actor_optimizer',
                        'critic_optimizer')

    def __init__(self, model, actor_optimizer, critic_optimizer, replay_buffer,
                 gamma, explorer,
                 gpu=None, replay_start_size=50000,
                 minibatch_size=32, update_interval=1,
                 target_update_interval=10000,
                 phi=lambda x: x,
                 target_update_method='hard',
                 soft_update_tau=1e-2,
                 n_times_update=1, average_q_decay=0.999,
                 average_loss_decay=0.99,
                 episodic_update=False,
                 episodic_update_len=None,
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
        if episodic_update:
            update_func = self.update_from_episodes
        else:
            update_func = self.update
        self.replay_updater = ReplayUpdater(
            replay_buffer=replay_buffer,
            update_func=update_func,
            batchsize=minibatch_size,
            episodic_update=episodic_update,
            episodic_update_len=episodic_update_len,
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

    # Update Q-function
    def compute_critic_loss(self, batch):
        """Compute loss for critic.

        Preconditions:
          target_q_function must have seen up to s_t and a_t.
          target_policy must have seen up to s_t.
          q_function must have seen up to s_{t-1}.
        Postconditions:
          target_q_function must have seen up to s_{t+1} and a_{t+1}.
          target_policy must have seen up to s_{t+1}.
          q_function must have seen up to s_t.
        """

        batch_next_state = batch['next_state']
        batch_rewards = batch['reward']
        batch_terminal = batch['is_state_terminal']
        batch_state = batch['state']
        batch_actions = batch['action']
        batch_next_actions = batch['next_action']
        batchsize = len(batch_rewards)

        with chainer.no_backprop_mode():
            # Target policy observes s_{t+1}
            next_actions = self.target_policy(
                batch_next_state).sample()

            # Q(s_{t+1}, mu(a_{t+1})) is evaluated.
            # This should not affect the internal state of Q.
            with state_kept(self.target_q_function):
                next_q = self.target_q_function(batch_next_state, next_actions)

            # Target Q-function observes s_{t+1} and a_{t+1}
            if isinstance(self.target_q_function, Recurrent):
                self.target_q_function.update_state(
                    batch_next_state, batch_next_actions)

            target_q = batch_rewards + self.gamma * \
                (1.0 - batch_terminal) * F.reshape(next_q, (batchsize,))

        # Estimated Q-function observes s_t and a_t
        predict_q = F.reshape(
            self.q_function(batch_state, batch_actions),
            (batchsize,))

        loss = F.mean_squared_error(target_q, predict_q)

        # Update stats
        self.average_critic_loss *= self.average_loss_decay
        self.average_critic_loss += ((1 - self.average_loss_decay) *
                                     float(loss.data))

        return loss

    def compute_actor_loss(self, batch):
        """Compute loss for actor.

        Preconditions:
          q_function must have seen up to s_{t-1} and s_{t-1}.
          policy must have seen up to s_{t-1}.
        Preconditions:
          q_function must have seen up to s_t and s_t.
          policy must have seen up to s_t.
        """

        batch_state = batch['state']
        batch_action = batch['action']
        batch_size = len(batch_action)

        # Estimated policy observes s_t
        onpolicy_actions = self.policy(batch_state).sample()

        # Q(s_t, mu(s_t)) is evaluated.
        # This should not affect the internal state of Q.
        with state_kept(self.q_function):
            q = self.q_function(batch_state, onpolicy_actions)

        # Estimated Q-function observes s_t and a_t
        if isinstance(self.q_function, Recurrent):
            self.q_function.update_state(batch_state, batch_action)

        # Avoid the numpy #9165 bug (see also: chainer #2744)
        q = q[:, :]

        # Since we want to maximize Q, loss is negation of Q
        loss = - F.sum(q) / batch_size

        # Update stats
        self.average_actor_loss *= self.average_loss_decay
        self.average_actor_loss += ((1 - self.average_loss_decay) *
                                    float(loss.data))
        return loss

    def update(self, experiences, errors_out=None):
        """Update the model from experiences"""

        batch = batch_experiences(experiences, self.xp, self.phi)
        self.critic_optimizer.update(lambda: self.compute_critic_loss(batch))
        self.actor_optimizer.update(lambda: self.compute_actor_loss(batch))

    def update_from_episodes(self, episodes, errors_out=None):
        # Sort episodes desc by their lengths
        sorted_episodes = list(reversed(sorted(episodes, key=len)))
        max_epi_len = len(sorted_episodes[0])

        # Precompute all the input batches
        batches = []
        for i in range(max_epi_len):
            transitions = []
            for ep in sorted_episodes:
                if len(ep) <= i:
                    break
                transitions.append(ep[i])
            batch = batch_experiences(
                transitions, xp=self.xp, phi=self.phi)
            batches.append(batch)

        with self.model.state_reset():
            with self.target_model.state_reset():

                # Since the target model is evaluated one-step ahead,
                # its internal states need to be updated
                self.target_q_function.update_state(
                    batches[0]['state'], batches[0]['action'])
                self.target_policy(batches[0]['state'])

                # Update critic through time
                critic_loss = 0
                for batch in batches:
                    critic_loss += self.compute_critic_loss(batch)
                self.critic_optimizer.update(lambda: critic_loss / max_epi_len)

        with self.model.state_reset():

            # Update actor through time
            actor_loss = 0
            for batch in batches:
                actor_loss += self.compute_actor_loss(batch)
            self.actor_optimizer.update(lambda: actor_loss / max_epi_len)

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

    def get_statistics(self):
        return [
            ('average_q', self.average_q),
            ('average_actor_loss', self.average_actor_loss),
            ('average_critic_loss', self.average_critic_loss),
        ]
