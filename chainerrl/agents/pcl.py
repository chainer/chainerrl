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
        t_max (int > 1 or None): Maximum length of trajectories sampled from
            the replay buffer. If set to None, there is not limit on it,
            complete trajectories / episodes will be sampled. Refer to the
            behavior of AbstractEpisodicReplayBuffer for more details.
        gamma (float): Discount factor [0,1]
        tau (float): Weight coefficient for the entropy regularizaiton term.
        phi (callable): Feature extractor function
        pi_loss_coef (float): Weight coefficient for the loss of the policy
        v_loss_coef (float): Weight coefficient for the loss of the value
            function
        rollout_len (int): Number of rollout steps (for computing path
            consistency, noted as d in the paper)
        batchsize (int): Number of episodes or sub-trajectories used for an
            update. The total number of transitions used will be
            (batchsize x t_max).
        disable_online_update (bool): If set true, disable online on-policy
            update and rely only on experience replay.
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
        backprop_future_values (bool): If set True, value gradients are
            computed not only wrt V(s_t) but also V(s_{t+d}).
        train_async (bool): If set True, use a process-local model to compute
            gradients and update the globally shared model.
    """

    process_idx = None
    saved_attributes = ['model', 'optimizer']
    shared_attributes = ['shared_model', 'optimizer']

    def __init__(self, model, optimizer,
                 replay_buffer=None,
                 t_max=None,
                 gamma=0.99,
                 tau=1e-2,
                 phi=lambda x: x,
                 pi_loss_coef=1.0,
                 v_loss_coef=0.5,
                 rollout_len=10,
                 batchsize=1,
                 disable_online_update=False,
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
                 backprop_future_values=True,
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
        if t_max is not None:
            assert t_max > 1, "t_max should be > 1, found %d" % t_max
        self.t_max = t_max
        self.gamma = gamma
        self.tau = tau
        self.phi = phi
        self.pi_loss_coef = pi_loss_coef
        self.v_loss_coef = v_loss_coef
        self.rollout_len = rollout_len
        if not self.xp.isscalar(batchsize):
            batchsize = self.xp.int32(batchsize)
            """Fix Chainer Issue #2807

            batchsize should (look to) be scalar.
            """
        self.batchsize = batchsize
        self.normalize_loss_by_steps = normalize_loss_by_steps
        self.act_deterministically = act_deterministically
        self.disable_online_update = disable_online_update
        self.n_times_replay = n_times_replay
        self.replay_start_size = replay_start_size
        self.average_loss_decay = average_loss_decay
        self.average_value_decay = average_value_decay
        self.average_entropy_decay = average_entropy_decay
        self.logger = logger if logger else getLogger(__name__)
        self.batch_states = batch_states
        self.backprop_future_values = backprop_future_values
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

    def _compute_path_consistency(self, v, last_v, rewards, log_probs, ts_max):
        """Compute squared soft consistency error on a sub-trajectory

        Args:
            v: Variable
            last_v: Variable
            rewards: list of arrays or Variables (single-element or vector)
            log_probs: list of Variables (single-element or vector)
            ts_max: list of integers indicating the length of episode(s)
        Returns:
            loss: Variable
        """

        # Get the length of the sub-trajectory
        d = len(rewards)
        assert 1 <= d <= self.rollout_len
        assert len(log_probs) == len(rewards)

        # Discounted sum of immediate rewards
        R_seq = chainerrl.functions.weighted_sum_arrays(
            xs=rewards,
            weights=[self.gamma ** i for i in range(d)])
        R_seq = F.expand_dims(R_seq, -1)

        # Discounted sum of log likelihoods
        G = chainerrl.functions.weighted_sum_arrays(
            xs=log_probs,
            weights=[self.gamma ** i for i in range(d)])
        G = F.expand_dims(G, -1)

        if not self.backprop_future_values:
            last_v = chainer.Variable(last_v.data)

        # Adapt the computation to different length of trajectories
        coef = self.xp.asarray([self.gamma ** t_max for t_max in ts_max])

        # C_pi only backprop through pi
        C_pi = (- v.data +
                coef * last_v.data +
                R_seq -
                self.tau * G)

        # C_v only backprop through v
        C_v = (- v +
               coef * last_v +
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

    def compute_loss(self, episodes, weights):
        """Compute gradients on a list of episodes

        If the episode's length, T, is larger than self.rollout_len,
        sub-trajectories will be used.

        The gradient is computed and accumulated on the fly.

        ############ Illustration of algorithm #############

        Given 2 (self.batchsize == 2) episodes of different length:

             0 1 2 3 4 5 6 7 8 9 (index, i)
        e_0  o o o o o o o o _ _ (o for transition, _ is fictive)
        e_1  o o o o o o _ _ _ _

        self.rollout_len == 4
        max_len == 7

        The algorithm will compute path consistency for all sub-trajectories
        of length self.rollout_len plus some shorter sub-trajectories at the
        head and the tail of the batch.

            i == 0: not enough transitions to compute the loss

                 0
            e_0  o
            e_1  o

            i == 1: compute loss with rollout_length = [1, 1]
            v is at position 0
            v_last is at 1 (rollout_length can be smaller than
                self.rollout_len) rewards and log_probs are at 0

                 0 1
            e_0  o o
            e_1  o o

            i == 4: compute loss with rollout_length = [4, 4]
            v is still at position 0
            v_last is at 4 (rollout_len == 4)
            rewards and log_probs are at 0, 1, 2, 3

                 0 1 2 3 4
            e_0  o o o o o
            e_1  o o o o o

            i == 5: compute loss with rollout_length = [4, 4]
            v is now at position 1
            v_last is at 5
            rewards and log_probs are at 1, 2, 3, 4

                 0 1 2 3 4 5
            e_0  x o o o o o (x means removed from the list like vs)
            e_1  x o o o o o

            i == 6: compute loss with rollout_length = [4, 3]
            v_last is at 6 (the last state is replicate for the second
                episode)
            rewards and log_probs are at 2, 3, 4, 5: at position 5, the
                reward and log_prob will be set to 0 as they all exceed
                the length of the episode. By setting them to 0, they
                will not contribute to the path consistency computation
            rollout_length: the second one only has a rollout length of
                3, because the length has been reached. This will tell
                the _compute_path_consistency how to compute the reduction
                coefficient for v_last

                 0 1 2 3 4 5 6
            e_0  x x o o o o o (x means removed from the list like vs)
            e_1  x x o o o o _

            i == 8: compute loss with rollout_length = [3, 1]

                 0 1 2 3 4 5 6 7 8
            e_0  x x x x o o o o _
            e_1  x x x x o o _ _ _ (removed, nb_episodes == 1)

            i == 9: compute loss with rollout_length = [2]
            The second episode now has too less transitions, no loss
                will be computed on that episode. In order to save computation,
                the batch size is reduced.

                 0 1 2 3 4 5 6 7 8 9
            e_0  x x x x x o o o _ _
            e_1  x x x x x o _ _ _ _ (removed, nb_episodes == 1 at this round)


        ####################################################

        :param episodes: list of list of length
        :param weights: list of weights
        :return: single value array
        """

        episodes = list(reversed(sorted(episodes, key=len)))
        max_len = len(episodes[0])
        nb_episodes = len(episodes)

        # Normalize with the length of the episode
        weights = [weights[j] / len(episodes[j])
                   for j in range(nb_episodes)]

        # Check the length of shortest episode
        while nb_episodes > 0 and len(episodes[-1]) < 2:
            nb_episodes -= 1
            episodes = episodes[:-1]
        # If no episode satisfies the constraints, do not update the model
        if nb_episodes == 0:
            return self.xp.asarray([0], dtype=self.xp.float32)

        # Save one more item
        vs = []
        rewards = []
        log_probs = []

        losses = []

        rollout_length = [-1 for _ in range(nb_episodes)]

        for i in range(max_len + self.rollout_len - 1):
            # Update minibatch size
            while len(episodes[-1]) + self.rollout_len - 2 < i:
                nb_episodes -= 1
                episodes = episodes[:-1]
                rollout_length = rollout_length[:-1]
                assert nb_episodes == len(episodes) == len(rollout_length)

            for j in range(nb_episodes):
                if i < len(episodes[j]):
                    rollout_length[j] += 1

            assert nb_episodes > 0

            for k in range(len(vs)):
                vs[k] = vs[k][:nb_episodes]
                rewards[k] = rewards[k][:nb_episodes]
                log_probs[k] = log_probs[k][:nb_episodes]

            # Replicate the last transition
            state = batch_states(
                [episode[i]['state'] if i < len(episode)
                 else episode[-1]['state'] for episode in episodes],
                self.xp, self.phi)

            # By default, the last transition correspond to the terminal
            # transition where the action/reward are meaningless.
            action = self.xp.asarray(
                [episode[i]['action'] if i < len(episode) - 1
                 else 0 for episode in episodes]).astype(self.xp.int8)
            # Compute pi and v
            action_distrib, v = self.model(state)

            # Use the action to get its own log-probability
            log_prob = action_distrib.log_prob(action)
            # Use zero is the length of episode is exceeded
            log_prob = F.where(self.xp.asarray([
                i < len(episode) - 1 for episode in episodes
            ]), log_prob, self.xp.zeros_like(log_prob).astype(self.xp.float32))

            log_probs.append(log_prob)

            # The size of reward decide the length of trajectory
            rewards.append(self.xp.asarray(
                [episode[i]['reward'] if i < len(episode) - 1
                 else 0 for episode in episodes], dtype=self.xp.float32))

            vs.append(v)

            # Pop the oldest element as in a deque
            if len(vs) > self.rollout_len + 1:
                vs = vs[1:]
                log_probs = log_probs[1:]
                rewards = rewards[1:]

                for j in range(nb_episodes):
                    rollout_length[j] -= 1

            if len(vs) > 1:
                # For debug and understanding of the algorithm
                for j in range(nb_episodes):
                    assert rollout_length[j] == \
                           min(i,
                               self.rollout_len,
                               len(episodes[j]) - 1 - (i - self.rollout_len),
                               len(episodes[j]) - 1)

                loss = self._compute_path_consistency(
                    vs[0],
                    vs[-1],
                    rewards[:-1],
                    log_probs[:-1],
                    rollout_length
                )

                if self.normalize_loss_by_steps:
                    # Normalize with the rollout length
                    w = self.xp.asarray([weights[j] / rollout_length[j]
                                         for j in range(nb_episodes)])
                else:
                    w = self.xp.asarray(weights[:nb_episodes])

                loss *= w

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

            # Compute loss on sampled trajectories
            loss = self.compute_loss(episodes, weights)

            self.update(loss)

    def update_on_policy(self):
        self.model.zerograds()

        # Compute the loss on the current episode
        with state_reset(self.model):
            loss = self.compute_loss([self.past_transitions], [1])
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
