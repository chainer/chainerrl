from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import copy
from logging import getLogger

import numpy as np

import chainer
from chainer import cuda
import chainer.functions as F

from chainerrl.agents import DoubleDQN
from chainerrl.agents.dqn import compute_value_loss
from chainerrl.agents.dqn import compute_weighted_value_loss

from chainerrl.recurrent import state_kept
from chainerrl.misc.batch_states import batch_states
from chainerrl.replay_buffer import batch_experiences
from chainerrl.replay_buffer import ReplayUpdater


class DemoReplayUpdater(object):
    """Object that handles update schedule and configurations.

    Args:
        demo_replay_buffer (ReplayBuffer): Replay buffer for demonstrations
        replay_buffer (ReplayBuffer): Replay buffer for self-play
        demo_sample_ratio (float): Sampling ratio from demonstration buffer
        update_func (callable): Callable that accepts one of these:
            (1) two lists of transition dicts (if episodic_update=False)
            (2) two lists of transition dicts (if episodic_update=True)
        replay_start_size (int): if the replay buffer's size is less than
            replay_start_size, skip update
        batchsize (int): Minibatch size
        update_interval (int): Model update interval in step
        n_times_update (int): Number of repetition of update
        episodic_update (bool): Use full episodes for update if set True
        episodic_update_len (int or None): Subsequences of this length are used
            for update if set int and episodic_update=True
    """

    def __init__(self, demo_replay_buffer, replay_buffer, demo_sample_ratio,
                 update_func, batchsize, episodic_update,
                 n_times_update, replay_start_size, update_interval,
                 episodic_update_len=None):

        assert batchsize <= replay_start_size
        self.minibatch_n_demos = int(demo_sample_ratio * float(batchsize))
        self.minibatch_n_rl = batchsize - self.minibatch_n_demos
        self.demo_replay_buffer = demo_replay_buffer
        self.replay_buffer = replay_buffer
        self.update_func = update_func
        self.batchsize = batchsize
        self.episodic_update = episodic_update
        self.episodic_update_len = episodic_update_len
        self.n_times_update = n_times_update
        self.replay_start_size = replay_start_size
        self.update_interval = update_interval

    def update_if_necessary(self, iteration):
        """Called during normal self-play
        """
        if len(self.replay_buffer) < self.replay_start_size:
            return

        if (self.episodic_update
                and self.replay_buffer.n_episodes < self.batchsize):
            return

        if iteration % self.update_interval != 0:
            return

        for _ in range(self.n_times_update):
            if self.episodic_update:
                episodes_rl = self.replay_buffer.sample_episodes(
                    self.batchsize, self.episodic_update_len)
                episodes_demo = self.demo_replay_buffer.sample_episodes(
                    self.batch_size, self.episodic_update_len)
                self.update_func(episodes_rl, episodes_demo)
            else:
                transitions_demo = self.demo_replay_buffer.sample(
                    self.minibatch_n_demos)
                transitions_rl = self.replay_buffer.sample(self.minibatch_n_rl)
                self.update_func(transitions_rl, transitions_demo)

    def update_from_demonstrations(self):
        """Called during pre-train steps. All samples are from demo buffer
        """
        if self.episodic_update:
            episodes_demo = self.demo_replay_buffer.sample_episodes(
                self.batch_size, self.episodic_update_len)
            self.update_func([], episodes_demo)
        else:
            transitions_demo = self.demo_replay_buffer.sample(self.batchsize)
            self.update_func([], transitions_demo)


class DQfD(DoubleDQN):
    """Deep-Q Learning from Demonstrations
    See: https://arxiv.org/abs/1704.03732.

    TODO:
        * Double DQN 1-step Update
        * Test batch observe & train.
        * Test episodic update


    Deviations from paper:
        * Fixed proportional sampling from the two replay buffers instead of
        single buffer with bonus priority for demos.

    DQN Args:
        q_function (StateQFunction): Q-function
        optimizer (Optimizer): Optimizer that is already setup
        replay_buffer (ReplayBuffer): Replay buffer
        gamma (float): Discount factor
        explorer (Explorer): Explorer that specifies an exploration strategy.
        gpu (int): GPU device id if not None nor negative.
        replay_start_size (int): if the replay buffer's size is less than
            replay_start_size, skip update
        minibatch_size (int): Minibatch size
        update_interval (int): Model update interval in step
        target_update_interval (int): Target model update interval in step
        clip_delta (bool): Clip delta if set True
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

    DQfD-specific args:
        n_pretrain_steps: Number of pretraining steps to perform
        demo_replay_buffer (ReplayBuffer): Persistent buffer for expert demonstrations
        demo_sample_ratio (float): Ratio of training minibatch from demos
        demo_supervised_margin (float): Margin width for supervised demo loss
        loss_coeff_supervised (float): Coefficient for the supervised loss term
        loss_coeff_l2 (float): Coefficient used to regulate weight decay rate
    """

    def __init__(self, q_function, optimizer,
                 replay_buffer, demo_replay_buffer,
                 gamma, explorer, n_pretrain_steps,
                 demo_supervised_margin=0.8,
                 loss_coeff_supervised=1.0,
                 loss_coeff_l2=1e-5,
                 demo_sample_ratio=0.2, gpu=None,
                 replay_start_size=50000,
                 minibatch_size=32, update_interval=1,
                 target_update_interval=10000, clip_delta=True,
                 phi=lambda x: x,
                 target_update_method='hard',
                 soft_update_tau=1e-2,
                 n_times_update=1, average_q_decay=0.999,
                 average_loss_decay=0.99,
                 batch_accumulator='mean', episodic_update=False,
                 episodic_update_len=None,
                 logger=getLogger(__name__),
                 batch_states=batch_states):

        super(DQfD, self).__init__(q_function, optimizer, replay_buffer, gamma,
                                   explorer, gpu, replay_start_size,
                                   minibatch_size, update_interval,
                                   target_update_interval, clip_delta,
                                   phi, target_update_method, soft_update_tau,
                                   n_times_update, average_q_decay,
                                   average_loss_decay, batch_accumulator,
                                   episodic_update, episodic_update_len,
                                   logger, batch_states)

        self.minibatch_size = minibatch_size
        self.n_pretrain_steps = n_pretrain_steps
        self.demo_supervised_margin = demo_supervised_margin
        self.loss_coeff_supervised = loss_coeff_supervised
        self.loss_coeff_l2 = loss_coeff_l2
        self.demo_replay_buffer = demo_replay_buffer

        self.optimizer.add_hook(
            chainer.optimizer_hooks.WeightDecay(loss_coeff_l2))

        # Overwrite DQN's replay updater.
        # TODO: Is there a better way to do this?

        self.replay_updater = DemoReplayUpdater(
            demo_replay_buffer=demo_replay_buffer,
            replay_buffer=replay_buffer,
            demo_sample_ratio=demo_sample_ratio,
            update_func=self.combined_loss,
            batchsize=minibatch_size,
            episodic_update=episodic_update,
            episodic_update_len=episodic_update_len,
            n_times_update=n_times_update,
            replay_start_size=replay_start_size,
            update_interval=update_interval,
        )

        # TODO: Should this really go here? Move intro train function?
        self.pretrain()

    def pretrain(self):
        """Uses expert demonstrations to does pre-training
        """
        for tpre in range(self.n_pretrain_steps):
            # The whole batch can consist of demo transitions in pretrain
            self.replay_updater.update_from_demonstrations()
            if tpre % self.target_update_interval == 0:
                self.sync_target_network()

    def update(self):
        """Invalidate DQN's update()
        DQfD's update happens via combined_loss()
        """
        raise NotImplementedError("update() is not valid for DQfD")

    def _compute_y_and_t(self, exp_batch):
        """Overwrites DQN's method
        """
        batch_size = exp_batch['reward'].shape[0]

        # Compute Q-values for current states
        batch_state = exp_batch['state']
        qout = self.model(batch_state)

        # Caches Q(s) for use in supervised demo loss
        self.qout = qout

        batch_actions = exp_batch['action']
        batch_q = F.reshape(qout.evaluate_actions(
            batch_actions), (batch_size, 1))

        with chainer.no_backprop_mode():
            batch_q_target = F.reshape(
                self._compute_target_values(exp_batch),
                (batch_size, 1))

        return batch_q, batch_q_target

    def combined_loss(self, experiences_rl, experiences_demo):
        """Combined DQfD loss function for Demonstration and self-play/RL.
        """
        num_exp_rl = len(experiences_rl)
        experiences = experiences_rl+experiences_demo
        exp_batch = batch_experiences(experiences, xp=self.xp, phi=self.phi,
                                      gamma=self.gamma,
                                      batch_states=self.batch_states)

        exp_batch['weights'] = self.xp.asarray(
            [elem[0]['weight']for elem in experiences], dtype=self.xp.float32)

        errors_out = []
        qloss_nstep = self._compute_loss(exp_batch, errors_out=errors_out)

        # Update priorities
        self.demo_replay_buffer.update_errors(errors_out[num_exp_rl:])
        if num_exp_rl > 0:
            self.replay_buffer.update_errors(errors_out[:num_exp_rl])

        # Large-margin supervised loss
        # Grab the cached Q(s) in the forward pass & subset demo exp.
        q_picked = self.qout.evaluate_actions(exp_batch["action"])
        q_expert_demos = q_picked[num_exp_rl:]

        # unwrap DiscreteActionValue and subset demos
        q_demos = self.qout.q_values[num_exp_rl:]

        # Calculate margin forall actions (l(a_E,a) in the paper)
        margin = np.zeros_like(q_demos.array) + self.demo_supervised_margin
        a_expert_demos = exp_batch["action"][num_exp_rl:]
        margin[np.arange(len(experiences_demo)), a_expert_demos] = 0.0

        supervised_targets = F.max(q_demos + margin, axis=-1)
        loss_supervised = F.sum(supervised_targets - q_expert_demos)
        if self.batch_accumulator is "mean":
            loss_supervised /= len(experiences_demo)

        # L2 Loss
        # TODO: Is there a better way to do this?
        # loss_l2 = chainer.Variable(np.zeros(1,dtype=np.float32))
        # for param in self.model.params():
            # flatparam = param.reshape(-1)
            # loss_l2 += F.matmul(flatparam, flatparam.T)

        loss_combined = qloss_nstep + \
            self.loss_coeff_supervised * loss_supervised
        # L2 loss is directly applied as chainer optimizer hook.
        # loss_combined += self.loss_coeff_l2 * loss_l2

        self.model.cleargrads()
        loss_combined.backward()
        self.optimizer.update()

        # Update stats
        self.average_loss *= self.average_loss_decay
        self.average_loss += (1 - self.average_loss_decay) * \
            float(loss_combined.array)

    def act_and_train(self, obs, reward):

        with chainer.using_config('train', False), chainer.no_backprop_mode():
            action_value = self.model(
                self.batch_states([obs], self.xp, self.phi))
            q = float(action_value.max.array)
            greedy_action = cuda.to_cpu(action_value.greedy_actions.array)[0]

        # Update stats
        self.average_q *= self.average_q_decay
        self.average_q += (1 - self.average_q_decay) * q

        self.logger.debug('t:%s q:%s action_value:%s', self.t, q, action_value)

        action = self.explorer.select_action(
            self.t, lambda: greedy_action, action_value=action_value)
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
                next_state=obs,
                next_action=action,
                is_state_terminal=False)

        self.last_state = obs
        self.last_action = action

        self.replay_updater.update_if_necessary(self.t)
        self.logger.debug('t:%s r:%s a:%s', self.t, reward, action)

        return self.last_action

    def batch_observe_and_train(self, batch_obs, batch_reward,
                                batch_done, batch_reset):
        for i in range(len(batch_obs)):
            self.t += 1
            # Update the target network
            if self.t % self.target_update_interval == 0:
                self.sync_target_network()
            if self.batch_last_obs[i] is not None:
                assert self.batch_last_action[i] is not None
                # Add a transition to the replay buffer
                self.replay_buffer.append(
                    state=self.batch_last_obs[i],
                    action=self.batch_last_action[i],
                    reward=batch_reward[i],
                    next_state=batch_obs[i],
                    next_action=None,
                    is_state_terminal=batch_done[i],
                    env_id=i,
                )
                if batch_reset[i] or batch_done[i]:
                    self.batch_last_obs[i] = None
                    self.replay_buffer.stop_current_episode(env_id=i)
            self.replay_updater.update_if_necessary(self.t)
