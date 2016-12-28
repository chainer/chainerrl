from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
from builtins import super
standard_library.install_aliases()

import chainer
import chainer.functions as F
from chainer import cuda

from chainerrl.agents import dqn
from chainerrl.recurrent import Recurrent
from chainerrl.recurrent import RecurrentChainMixin


class DDPGModel(chainer.Chain, RecurrentChainMixin):

    def __init__(self, policy, q_func):
        super().__init__(policy=policy, q_function=q_func)


class DDPG(dqn.DQN):
    """Deep Deterministic Policy Gradients.

    Args:
        model (DDPGModel): DDPG model that contains both a policy and a
            Q-function
        actor_optimizer (Optimizer): Optimizer setup with the policy
        critic_optimizer (Optimizer): Optimizer setup with the Q-function

        For other arguments, see DQN
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

        # Target policy observes s_{t+1}
        next_actions = self.target_policy(batch_next_state, test=True).sample()

        # Q(s_{t+1}, mu(a_{t+1})) is evaluated.
        # This should not affect the internal state of Q.
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
        Preconditions:
          q_function must have seen up to s_{t-1} and s_{t-1}.
          policy must have seen up to s_{t-1}.
        Preconditions:
          q_function must have seen up to s_t and s_t.
          policy must have seen up to s_t.
        """

        batch_state = batch['state']
        batch_action = batch['action']
        batch_size = batch_state.shape[0]

        # Estimated policy observes s_t
        onpolicy_actions = self.policy(batch_state, test=False).sample()

        # Q(s_t, mu(s_t)) is evaluated.
        # This should not affect the internal state of Q.
        if isinstance(self.q_function, Recurrent):
            with self.q_function.state_kept():
                q = self.q_function(batch_state, onpolicy_actions, test=True)
        else:
            q = self.q_function(batch_state, onpolicy_actions, test=True)

        # import copy
        # q = copy.deepcopy(self.q_function)(batch_state, onpolicy_actions, test=True)

        # Estimated Q-function observes s_t and a_t
        self.q_function.update_state(batch_state, batch_action, test=False)

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
            batch = dqn.batch_experiences(
                transitions, xp=self.xp, phi=self.phi)
            batches.append(batch)

        with self.model.state_reset():
            with self.target_model.state_reset():

                # Since the target model is evaluated one-step ahead,
                # its internal states need to be updated
                self.input_initial_batch_target_model(batches[0])

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

    def act(self, state):

        s = self._batch_states([state])
        action = self.policy(s, test=True).sample()
        # Q is not needed here, but log it just for information
        q = self.q_function(s, action, test=True)

        # Update stats
        self.average_q *= self.average_q_decay
        self.average_q += (1 - self.average_q_decay) * float(q.data)

        self.logger.debug('t:%s a:%s q:%s',
                          self.t, action.data[0], q.data)
        return cuda.to_cpu(action.data[0])

    @property
    def saved_attributes(self):
        return ('model', 'target_model', 'actor_optimizer', 'critic_optimizer')

    def get_stats_keys(self):
        return ('average_q', 'average_actor_loss', 'average_critic_loss')

    def get_stats_values(self):
        return (self.average_q, self.average_actor_loss, self.average_critic_loss)

    def input_initial_batch_target_model(self, batch):
        self.target_q_function.update_state(
            batch['state'], batch['action'], test=True)
        self.target_policy(batch['state'])
