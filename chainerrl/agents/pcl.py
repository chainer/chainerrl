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
from chainer import Variable

import chainerrl
from chainerrl import agent
from chainerrl.agents import a3c
from chainerrl.misc import async
from chainerrl.misc.batch_states import batch_states
from chainerrl.misc import copy_param
from chainerrl.recurrent import Recurrent
from chainerrl.recurrent import state_reset


def asfloat(x):
    if isinstance(x, chainer.Variable):
        return float(x.data)
    else:
        return float(x)


PCLSeparateModel = a3c.A3CSeparateModel
PCLSharedModel = a3c.A3CSharedModel


class PCL(agent.AttributeSavingMixin, agent.AsyncAgent):
    """PCL (Path Consistency Learning).

    Not only the batch PCL algorithm proposed in the paper but also its
    asynchronous variant is implemented.

    See https://arxiv.org/abs/1702.08892

    Args:
        model (chainer.Link): Model to train. It must be a callable that
            accepts a batch of observations as input and return two values:

            - action distributions (Distribution)
            - state values (chainer.Variable)
        optimizer (chainer.Optimizer): optimizer used to train the model
        gamma (float): Discount factor [0,1]
        tau (float): Weight coefficient for the entropy regularizaiton term.
        phi (callable): Feature extractor function
        pi_loss_coef (float): Weight coefficient for the loss of the policy
        v_loss_coef (float): Weight coefficient for the loss of the value
            function
        rollout_len (int): Number of rollout steps
        batchsize (int): Number of episodes or sub-trajectories used for an
            update. The total number of transitions used will be
            (batchsize x t_max).
        disable_online_update (bool): If set true, disable online on-policy
            update and rely only on experience replay.
        t_max (int or None): Maximum length of trajectories sampled from the
            replay buffer. If set to None, there is not limit on it,
            complete trajectories / episodes will be sampled. Refer to the
            behavior of AbstractEpisodicReplayBuffer for more details.
        n_times_replay (int): Number of times experience replay is repeated per
            one time of online update.
        replay_start_size (int): Experience replay is disabled if the number of
            transitions in the replay buffer is lower than this value.
        normalize_loss_by_steps (bool): If set true, losses are normalized by
            the number of steps taken to accumulate the losses
        act_deterministically (bool): If set true, choose most probable actions
            in act method.
        average_loss_decay (float): Decay rate of average loss. Used only
            to record statistics.
        average_entropy_decay (float): Decay rate of average entropy. Used only
            to record statistics.
        average_value_decay (float): Decay rate of average value. Used only
            to record statistics.
        explorer (Explorer or None): If not None, this explorer is used for
            selecting actions.
        logger (None or Logger): Logger to be used
        batch_states (callable): Method which makes a batch of observations.
            default is `chainerrl.misc.batch_states.batch_states`
        train_async (bool): If set True, use a process-local model to compute
            gradients and update the globally shared model.
    """

    process_idx = None
    saved_attributes = ['model', 'optimizer']
    shared_attributes = ['shared_model', 'optimizer']

    def __init__(self, model, optimizer,
                 replay_buffer=None,
                 gamma=0.99,
                 tau=1e-2,
                 phi=lambda x: x,
                 pi_loss_coef=1.0,
                 v_loss_coef=0.5,
                 rollout_len=10,
                 batchsize=1,
                 disable_online_update=False,
                 t_max=None,
                 n_times_replay=1,
                 replay_start_size=10 ** 2,
                 normalize_loss_by_steps=True,
                 act_deterministically=False,
                 average_loss_decay=0.999,
                 average_entropy_decay=0.999,
                 average_value_decay=0.999,
                 explorer=None,
                 logger=None,
                 batch_states=batch_states,
                 train_async=False):

        if train_async:
            # Globally shared model
            self.shared_model = model

            # Thread specific model
            self.model = copy.deepcopy(self.shared_model)
            async.assert_params_not_shared(self.shared_model, self.model)
        else:
            self.model = model
        self.xp = self.model.xp

        self.optimizer = optimizer

        self.replay_buffer = replay_buffer
        self.gamma = gamma
        self.tau = tau
        self.phi = phi
        self.pi_loss_coef = pi_loss_coef
        self.v_loss_coef = v_loss_coef
        self.rollout_len = rollout_len
        self.batchsize = batchsize
        self.normalize_loss_by_steps = normalize_loss_by_steps
        self.act_deterministically = act_deterministically
        self.disable_online_update = disable_online_update
        self.t_max = t_max
        self.n_times_replay = n_times_replay
        self.replay_start_size = replay_start_size
        self.average_loss_decay = average_loss_decay
        self.average_value_decay = average_value_decay
        self.average_entropy_decay = average_entropy_decay
        self.logger = logger if logger else getLogger(__name__)
        self.batch_states = batch_states
        self.train_async = train_async

        self.t = 0
        self.t_local = 0
        self.explorer = explorer
        self.online_batch_losses = []

        # Stats
        self.average_loss = 0
        self.average_value = 0
        self.average_entropy = 0

        self.init_history_data_for_online_update()

    def init_history_data_for_online_update(self):
        # Use list to store the current episode
        self.past_transitions = []
        self.t_local = 0

    def sync_parameters(self):
        copy_param.copy_param(target_link=self.model,
                              source_link=self.shared_model)

    def _compute_path_consistency(self, v, last_v, rewards, log_probs):
        """Compute squared soft consistency error on a sub-trajectory"""

        # Get the length of the sub-trajectory
        d = len(rewards)
        assert 1 <= d <= self.rollout_len
        assert len(log_probs) == len(rewards)

        # Discounted sum of immediate rewards
        R_seq = sum(self.gamma ** i * rewards[i] for i in range(d))

        # Discounted sum of log likelihoods
        G = chainerrl.functions.weighted_sum_arrays(
            xs=log_probs,
            weights=[self.gamma ** i for i in range(d)])
        G = F.expand_dims(G, -1)

        # C_pi only backprop through pi
        C_pi = (- v.data +
                self.gamma ** d * last_v.data +
                R_seq -
                self.tau * G)

        # C_v only backprop through v
        C_v = (- v +
               self.gamma ** d * last_v +
               R_seq -
               self.tau * G.data)

        pi_loss = C_pi ** 2 / 2
        v_loss = C_v ** 2 / 2

        # Since we want to apply different learning rate, the computation of
        # C_pi and C_v must be done separately even though they are
        # numerically equal
        pi_loss *= self.pi_loss_coef
        v_loss *= self.v_loss_coef

        return pi_loss + v_loss

    def compute_loss(self, episode, weight=1):
        """Compute squared soft consistency error on the given trajectory

        If the episode's length, T, is larger than d (self.rollout_len),
        sub-trajectories will be used.

        The gradient is computed and accumulated on the fly.

        Args:
            episode: sequence of transitions (dict)
            weight: scalar

        Returns:
            loss over the trajectory as a scalar value
        """
        seq_len = len(episode)
        assert seq_len >= 1

        values = []
        rewards = [elem['reward'] for elem in episode]
        logs_probs = []

        # Process the trajectory in batches to accelerate
        for i in range((seq_len + self.batchsize - 1) // self.batchsize):
            # Get current batch size
            batchsize = min(seq_len - self.batchsize * i, self.batchsize)

            # Process the list of state for computation
            transitions = episode[self.batchsize * i:
                                  self.batchsize * i + batchsize]
            batch = {
                'state': batch_states(
                    [elem['state'] for elem in transitions],
                    self.xp, self.phi),
                'action': self.xp.asarray([elem['action']
                                           for elem in transitions]),
            }

            # Compute pi and v
            action_distrib, v = self.model(batch['state'])

            # Use the action to get its own log-probability
            logs_prob = action_distrib.log_prob(batch['action'])

            # Save the individual result (use slice and not one single index)
            for j in range(batchsize):
                values.append(v[j: j + 1])
                logs_probs.append(logs_prob[j: j + 1])

        assert len(values) == len(logs_probs) == len(rewards) == seq_len

        # Moving window
        losses = []
        for i in range(seq_len):
            d = min(seq_len - i, self.rollout_len)

            v = values[i]
            last_v = values[i + d] if i + d < seq_len \
                else Variable(self.xp.array([[0]], dtype=self.xp.float32))

            # Compute loss on a sub-trajectory
            loss = self._compute_path_consistency(v,
                                                  last_v,
                                                  rewards[i: i + d],
                                                  logs_probs[i: i + d])

            # Normalize with the length of the episode
            loss *= weight / seq_len

            if self.normalize_loss_by_steps:
                loss /= d

            # Compute gradient immediately
            loss.backward()

            # Save the value for logging purpose
            losses.append(loss.data)

        # Accumulate the loss
        loss = chainerrl.functions.sum_arrays(losses)

        if self.process_idx == 0:
            self.logger.debug('loss:%s', loss.data)

        return loss.data

    def update(self, loss):
        """Optimize the model

        This function calls the optimizer, but all the gradients should
        be already computed in other methods. This function also manages
        optimization over multiple processes.

        Args:
            loss as a array for logging purpose
        """

        self.average_loss += (
            (1 - self.average_loss_decay) *
            (asfloat(loss) - self.average_loss))

        if self.train_async:
            # Copy the gradients to the globally shared model
            self.shared_model.zerograds()
            copy_param.copy_grad(
                target_link=self.shared_model, source_link=self.model)
            if self.process_idx == 0:
                xp = self.xp
                norm = sum(xp.sum(xp.square(param.grad))
                           for param in self.optimizer.target.params())
                self.logger.debug('grad norm:%s', norm)
        self.optimizer.update()

        if self.train_async:
            self.sync_parameters()
        if isinstance(self.model, Recurrent):
            self.model.unchain_backward()

    def update_from_replay(self):
        if self.replay_buffer is None:
            return

        if len(self.replay_buffer) < self.replay_start_size:
            return

        if self.replay_buffer.n_episodes == 0:
            return

        if self.process_idx == 0:
            self.logger.debug('update_from_replay')

        # Clear the gradients in the model
        self.model.zerograds()

        with state_reset(self.model):
            # Sample one trajectory at a time
            episodes = self.replay_buffer.sample_episodes(
                1, max_len=self.t_max)
            if isinstance(episodes, tuple):
                # Prioritized replay
                episodes, weights = episodes
            else:
                weights = [1] * len(episodes)

            losses = []
            for i in range(len(episodes)):
                # Compute loss on a sampled trajectory
                loss = self.compute_loss(episodes[i], weights[i])
                losses.append(loss)

            self.update(chainerrl.functions.sum_arrays(losses))

    def update_on_policy(self):
        self.model.zerograds()

        # Compute the loss on the current episode
        with state_reset(self.model):
            loss = self.compute_loss(self.past_transitions)
            self.update(loss)

    def act_and_train(self, obs, reward):
        statevar = self.batch_states([obs], self.xp, self.phi)

        if len(self.past_transitions) > 0:
            self.past_transitions[-1]['reward'] = reward

        action_distrib, v = self.model(statevar)
        action = chainer.cuda.to_cpu(action_distrib.sample().data)[0]
        if self.explorer is not None:
            action = self.explorer.select_action(self.t, lambda: action)

        if len(self.past_transitions) > 0:
            # Add a transition to the replay buffer
            self.replay_buffer.append(
                state=self.past_transitions[-1]['state'],
                action=self.past_transitions[-1]['action'],
                reward=reward,
                is_state_terminal=False
            )

        # Save values for a later update
        self.past_transitions.append(dict(state=obs, action=action))

        # Increment global steps and local per episode steps
        self.t += 1
        self.t_local += 1

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

        return action

    def act(self, obs):
        # Use the process-local model for acting
        with chainer.no_backprop_mode():
            statevar = self.batch_states([obs], self.xp, self.phi)
            action_distrib, _ = self.model(statevar)
            if self.act_deterministically:
                return chainer.cuda.to_cpu(
                    action_distrib.most_probable.data)[0]
            else:
                return chainer.cuda.to_cpu(action_distrib.sample().data)[0]

    def stop_episode_and_train(self, state, reward, done=False):
        assert len(self.past_transitions) > 0

        self.past_transitions[-1]['reward'] = reward

        # The number of transitions should match the length of episode
        assert len(self.past_transitions) == self.t_local

        if not self.disable_online_update:
            self.update_on_policy()

        for _ in range(self.n_times_replay):
            self.update_from_replay()

        # Add a transition to the replay buffer
        self.replay_buffer.append(
            state=self.past_transitions[-1]['state'],
            action=self.past_transitions[-1]['action'],
            reward=reward,
            is_state_terminal=done)
        self.replay_buffer.stop_current_episode()

        # Clean up
        self.init_history_data_for_online_update()
        self.stop_episode()

    def stop_episode(self):
        if isinstance(self.model, Recurrent):
            self.model.reset_state()

    def load(self, dirname):
        super().load(dirname)
        if self.train_async:
            copy_param.copy_param(target_link=self.shared_model,
                                  source_link=self.model)

    def get_statistics(self):
        return [
            ('average_loss', self.average_loss),
            ('average_value', self.average_value),
            ('average_entropy', self.average_entropy),
        ]
