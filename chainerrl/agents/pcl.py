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
from chainerrl.recurrent import state_kept
from chainerrl.recurrent import state_reset
from chainerrl.replay_buffer import batch_experiences


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
        t_max (int or None): The model is updated after every t_max local
            steps. If set None, the model is updated after every episode.
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
        self.t_max = t_max
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
        self.last_state = None
        self.last_action = None
        self.explorer = explorer
        self.online_batch_losses = []

        # Stats
        self.average_loss = 0
        self.average_value = 0
        self.average_entropy = 0

        self.init_history_data_for_online_update()

    def init_history_data_for_online_update(self):
        self.past_actions = {}
        self.past_rewards = {}
        self.past_values = {}
        self.past_action_distrib = {}
        self.t_start = self.t

    def sync_parameters(self):
        copy_param.copy_param(target_link=self.model,
                              source_link=self.shared_model)

    def compute_loss(self, t_start, t_stop, rewards, values,
                     next_values, log_probs):

        seq_len = t_stop - t_start
        assert len(rewards) == seq_len
        assert len(values) == seq_len
        assert len(next_values) == seq_len
        assert len(log_probs) == seq_len

        pi_losses = []
        v_losses = []
        for t in range(t_start, t_stop):
            d = min(t_stop - t, self.rollout_len)
            # Discounted sum of immediate rewards
            R_seq = sum(self.gamma ** i * rewards[t + i] for i in range(d))
            # Discounted sum of log likelihoods
            G = chainerrl.functions.weighted_sum_arrays(
                xs=[log_probs[t + i] for i in range(d)],
                weights=[self.gamma ** i for i in range(d)])
            G = F.expand_dims(G, -1)
            last_v = next_values[t + d - 1]
            if not self.backprop_future_values:
                last_v = chainer.Variable(last_v.data)

            # C_pi only backprop through pi
            C_pi = (- values[t].data +
                    self.gamma ** d * last_v.data +
                    R_seq -
                    self.tau * G)

            # C_v only backprop through v
            C_v = (- values[t] +
                   self.gamma ** d * last_v +
                   R_seq -
                   self.tau * G.data)

            pi_losses.append(C_pi ** 2)
            v_losses.append(C_v ** 2)

        pi_loss = chainerrl.functions.sum_arrays(pi_losses) / 2
        v_loss = chainerrl.functions.sum_arrays(v_losses) / 2

        # Re-scale pi loss so that it is independent from tau
        pi_loss /= self.tau

        pi_loss *= self.pi_loss_coef
        v_loss *= self.v_loss_coef

        if self.normalize_loss_by_steps:
            pi_loss /= seq_len
            v_loss /= seq_len

        if self.process_idx == 0:
            self.logger.debug('pi_loss:%s v_loss:%s',
                              pi_loss.data, v_loss.data)

        return pi_loss + F.reshape(v_loss, pi_loss.data.shape)

    def update(self, loss):

        self.average_loss += (
            (1 - self.average_loss_decay) *
            (asfloat(loss) - self.average_loss))

        # Compute gradients using thread-specific model
        self.model.zerograds()
        loss.backward()
        if self.train_async:
            # Copy the gradients to the globally shared model
            self.shared_model.zerograds()
            copy_param.copy_grad(
                target_link=self.shared_model, source_link=self.model)
            if self.process_idx == 0:
                norm = self.optimizer.compute_grads_norm()
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

        if self.process_idx == 0:
            self.logger.debug('update_from_replay')

        episodes = self.replay_buffer.sample_episodes(
            self.batchsize, max_len=self.t_max)
        if isinstance(episodes, tuple):
            # Prioritized replay
            episodes, weights = episodes
        else:
            weights = [1] * len(episodes)
        sorted_episodes = list(reversed(sorted(episodes, key=len)))
        max_epi_len = len(sorted_episodes[0])

        with state_reset(self.model):
            # Batch computation of multiple episodes
            rewards = {}
            values = {}
            next_values = {}
            log_probs = {}
            next_action_distrib = None
            next_v = None
            for t in range(max_epi_len):
                transitions = []
                for ep in sorted_episodes:
                    if len(ep) <= t:
                        break
                    transitions.append(ep[t])
                batch = batch_experiences(transitions,
                                          xp=self.xp,
                                          phi=self.phi,
                                          batch_states=self.batch_states)
                batchsize = batch['action'].shape[0]
                if next_action_distrib is not None:
                    action_distrib = next_action_distrib[0:batchsize]
                    v = next_v[0:batchsize]
                else:
                    action_distrib, v = self.model(batch['state'])
                next_action_distrib, next_v = self.model(batch['next_state'])
                values[t] = v
                next_values[t] = next_v * \
                    (1 - batch['is_state_terminal'].reshape(next_v.shape))
                rewards[t] = chainer.cuda.to_cpu(batch['reward'])
                log_probs[t] = action_distrib.log_prob(batch['action'])
            # Loss is computed one by one episode
            losses = []
            for i, ep in enumerate(sorted_episodes):
                e_values = {}
                e_next_values = {}
                e_rewards = {}
                e_log_probs = {}
                for t in range(len(ep)):
                    assert values[t].shape[0] > i
                    assert next_values[t].shape[0] > i
                    assert rewards[t].shape[0] > i
                    assert log_probs[t].shape[0] > i
                    e_values[t] = values[t][i:i + 1]
                    e_next_values[t] = next_values[t][i:i + 1]
                    e_rewards[t] = float(rewards[t][i:i + 1])
                    e_log_probs[t] = log_probs[t][i:i + 1]
                losses.append(self.compute_loss(
                    t_start=0,
                    t_stop=len(ep),
                    rewards=e_rewards,
                    values=e_values,
                    next_values=e_next_values,
                    log_probs=e_log_probs))
            loss = chainerrl.functions.weighted_sum_arrays(
                losses, weights) / self.batchsize
            self.update(loss)

    def update_on_policy(self, statevar):
        assert self.t_start < self.t

        if not self.disable_online_update:
            next_values = {}
            for t in range(self.t_start + 1, self.t):
                next_values[t - 1] = self.past_values[t]
            if statevar is None:
                next_values[self.t - 1] = chainer.Variable(
                    self.xp.zeros_like(self.past_values[self.t - 1].data))
            else:
                with state_kept(self.model):
                    _, v = self.model(statevar)
                next_values[self.t - 1] = v
            log_probs = {t: self.past_action_distrib[t].log_prob(
                self.xp.asarray(self.xp.expand_dims(a, 0)))
                for t, a in self.past_actions.items()}
            self.online_batch_losses.append(self.compute_loss(
                t_start=self.t_start, t_stop=self.t,
                rewards=self.past_rewards,
                values=self.past_values,
                next_values=next_values,
                log_probs=log_probs))
            if len(self.online_batch_losses) == self.batchsize:
                loss = chainerrl.functions.sum_arrays(
                    self.online_batch_losses) / self.batchsize
                self.update(loss)
                self.online_batch_losses = []

        self.init_history_data_for_online_update()

    def act_and_train(self, state, reward):

        statevar = self.batch_states([state], self.xp, self.phi)

        if self.last_state is not None:
            self.past_rewards[self.t - 1] = reward

        if self.t - self.t_start == self.t_max:
            self.update_on_policy(statevar)
            if len(self.online_batch_losses) == 0:
                for _ in range(self.n_times_replay):
                    self.update_from_replay()

        action_distrib, v = self.model(statevar)
        action = chainer.cuda.to_cpu(action_distrib.sample().data)[0]
        if self.explorer is not None:
            action = self.explorer.select_action(self.t, lambda: action)

        # Save values for a later update
        self.past_values[self.t] = v
        self.past_actions[self.t] = action
        self.past_action_distrib[self.t] = action_distrib

        self.t += 1

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

        if self.last_state is not None:
            assert self.last_action is not None
            assert self.last_action_distrib is not None
            # Add a transition to the replay buffer
            self.replay_buffer.append(
                state=self.last_state,
                action=self.last_action,
                reward=reward,
                next_state=state,
                next_action=action,
                is_state_terminal=False,
                mu=self.last_action_distrib,
            )

        self.last_state = state
        self.last_action = action
        self.last_action_distrib = action_distrib.copy()

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
        assert self.last_state is not None
        assert self.last_action is not None

        self.past_rewards[self.t - 1] = reward
        if done:
            self.update_on_policy(None)
        else:
            statevar = self.batch_states([state], self.xp, self.phi)
            self.update_on_policy(statevar)
        if len(self.online_batch_losses) == 0:
            for _ in range(self.n_times_replay):
                self.update_from_replay()

        if isinstance(self.model, Recurrent):
            self.model.reset_state()

        # Add a transition to the replay buffer
        self.replay_buffer.append(
            state=self.last_state,
            action=self.last_action,
            reward=reward,
            next_state=state,
            next_action=self.last_action,
            is_state_terminal=done,
            mu=self.last_action_distrib)
        self.replay_buffer.stop_current_episode()

        self.last_state = None
        self.last_action = None
        self.last_action_distrib = None

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
