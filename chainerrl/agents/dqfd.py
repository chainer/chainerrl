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


class DQfD(DoubleDQN):
    """Deep-Q Learning from Demonstrations
    See: https://arxiv.org/abs/1704.03732.

    TODO:
        Fix logging of loss statistics

    Deviations from paper:
        * Fixed proportional sampling from the two replay buffers instead of
        single buffer with bonus priority for demos.
        * Only n_step update applied (1 or n. Not both)

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

        # Overwrite DQN's replay updater
        # TODO: Dirty. Find a better way.
        n_demos_sampled = int(float(minibatch_size)*demo_sample_ratio)
        self.replay_updater = ReplayUpdater(
            replay_buffer=replay_buffer,
            update_func=self.dqn_loss,
            batchsize=(minibatch_size-n_demos_sampled),
            episodic_update=episodic_update,
            episodic_update_len=episodic_update_len,
            n_times_update=n_times_update,
            replay_start_size=replay_start_size,
            update_interval=update_interval,
        )

        self.minibatch_size = minibatch_size
        self.n_pretrain_steps = n_pretrain_steps
        self.demo_replay_buffer = demo_replay_buffer
        self.demo_sample_ratio = demo_sample_ratio
        self.demo_supervised_margin = demo_supervised_margin
        self.loss_coeff_supervised = loss_coeff_supervised

        self.demo_replay_updater = ReplayUpdater(
            replay_buffer=demo_replay_buffer,
            update_func=self.demo_loss,
            batchsize=n_demos_sampled,
            episodic_update=episodic_update,
            episodic_update_len=episodic_update_len,
            n_times_update=n_times_update,
            replay_start_size=replay_start_size,
            update_interval=update_interval,
        )

        self.optimizer.add_hook(chainer.optimizer.WeightDecay(loss_coeff_l2))

        # TODO: Should this really go here?
        self.pretrain()

    def pretrain(self):
        """Loads expert demonstrations and does pre-training
        """
        # Gradients are accumulated over two fns. An initial clear is needed.
        self.model.cleargrads()

        for tpre in range(self.n_pretrain_steps):
            # The whole batch can consist of demo transitions in pretrain
            transitions = self.demo_replay_buffer.sample(self.minibatch_size)
            self.demo_loss(transitions)
            self.apply_grads()

            if tpre % self.target_update_interval == 0:
                self.sync_target_network()

    def update(self):
        """Invalidate DQN's update()
        DQfD's update happens via dqn_loss and demo_loss
        .. and then applied via the apply_grads() function
        """
        raise NotImplementedError("update() is not valid for DQfD")

    def _compute_target_values(self, exp_batch):
        batch_next_state = exp_batch['next_state']

        target_next_qout = self.target_model(batch_next_state)
        self.target_next_qout = target_next_qout
        next_q_max = target_next_qout.max

        batch_rewards = exp_batch['reward']
        batch_terminal = exp_batch['is_state_terminal']
        discount = exp_batch['discount']

        return batch_rewards + discount * (1.0 - batch_terminal) * next_q_max

    def dqn_loss(self, experiences, errors_out=None):
        """Calculate the gradients for the DQN update.
        Args:
            experiences (list): List of lists of dicts.
                For DQN, each dict must contains:
                  - state (object): State
                  - action (object): Action
                  - reward (float): Reward
                  - is_state_terminal (bool): True iff next state is terminal
                  - next_state (object): Next state
                  - weight (float, optional): Weight coefficient. It can be
                    used for importance sampling.
            errors_out (list or None): If set to a list, then TD-errors
                computed from the given experiences are appended to the list.
        Returns:
            None
        """
        exp_batch = batch_experiences(
            experiences, xp=self.xp,
            phi=self.phi, gamma=self.gamma,
            batch_states=self.batch_states)
        exp_batch['weights'] = self.xp.asarray(
                [elem[0]['weight']for elem in experiences],
                dtype=self.xp.float32)
        if errors_out is None:
                errors_out = []
        loss_q = self._compute_loss(exp_batch, errors_out=errors_out)
        self.replay_buffer.update_errors(errors_out)

        # Update stats
        self.average_loss *= self.average_loss_decay
        self.average_loss += (1 - self.average_loss_decay) * \
            float(loss_q.array)
        loss_q.backward()


    def _compute_y_and_t(self, exp_batch):
        batch_size = exp_batch['reward'].shape[0]

        # Compute Q-values for current states
        batch_state = exp_batch['state']

        qout = self.model(batch_state)

        # Cache Q(s) for use in supervised loss
        self.qout = qout

        batch_actions = exp_batch['action']
        batch_q = F.reshape(qout.evaluate_actions(
            batch_actions), (batch_size, 1))

        with chainer.no_backprop_mode():
            batch_q_target = F.reshape(
                self._compute_target_values(exp_batch),
                (batch_size, 1))

        return batch_q, batch_q_target

    def demo_loss(self, experiences):
        """Calculate the gradients for the demonstration update.
        Args:
            experiences (list): List of lists of dicts.
                For DQN, each dict must contains:
                  - state (object): State
                  - action (object): Action
                  - reward (float): Reward
                  - is_state_terminal (bool): True iff next state is terminal
                  - next_state (object): Next state
                  - weight (float, optional): Weight coefficient. It can be
                    used for importance sampling.
        Returns:
            None
        """
        # n-step Q-learning loss
        exp_batch = batch_experiences(experiences, xp=self.xp, phi=self.phi,
                                      gamma=self.gamma,
                                      batch_states=self.batch_states)
        exp_batch['weights'] = self.xp.asarray(
            [elem[0]['weight']for elem in experiences], dtype=self.xp.float32)

        errors_out = []
        loss_q_nstep = self._compute_loss(exp_batch, errors_out=errors_out)
        self.demo_replay_buffer.update_errors(errors_out)

        # Large-margin supervised loss
        a_expert = exp_batch['action']
        batch_size = exp_batch['reward'].shape[0]

        # Grab the cached Q(s)
        q_values = self.qout
        q_expert = q_values.evaluate_actions(a_expert)
        q_values = q_values.q_values.array  # unwrap DiscreteActionValue

        margin = np.zeros_like(q_values) + self.demo_supervised_margin
        margin[:, a_expert] = 0
        supervised_targets = np.max(q_values + margin, axis=-1)

        loss_supervised = F.mean_squared_error(supervised_targets, q_expert)

        loss_demo = loss_q_nstep + self.loss_coeff_supervised * loss_supervised

        loss_demo.backward()

        # Update stats
        self.average_loss *= self.average_loss_decay
        self.average_loss += (1 - self.average_loss_decay) * \
            float(loss_demo.array)

    def apply_grads(self):
        """Applies the accumulated gradients from dqn_loss and demo_loss
        """
        self.optimizer.update()
        self.model.cleargrads()

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

        self.demo_replay_updater.update_if_necessary(self.t)
        self.replay_updater.update_if_necessary(self.t)
        self.apply_grads()

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
            self.demo_replay_updater.update_if_necessary(self.t)
            self.apply_grads()
