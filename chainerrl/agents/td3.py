from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
from builtins import *  # NOQA
standard_library.install_aliases()  # NOQA

import collections
import copy
from logging import getLogger

import chainer
from chainer import cuda
import chainer.functions as F
import numpy as np

from chainerrl.agent import AttributeSavingMixin
from chainerrl.agent import BatchAgent
from chainerrl.misc.batch_states import batch_states
from chainerrl.misc.copy_param import synchronize_parameters
from chainerrl.replay_buffer import batch_experiences
from chainerrl.replay_buffer import ReplayUpdater


def _mean_or_nan(xs):
    """Return its mean a non-empty sequence, numpy.nan for a empty one."""
    return np.mean(xs) if xs else np.nan


def default_target_policy_smoothing_func(batch_action):
    """Add noises to actions for target policy smoothing."""
    xp = cuda.get_array_module(batch_action)
    noise = xp.clip(xp.random.normal(
        loc=0, scale=0.2, size=batch_action.shape).astype(
            batch_action.dtype), -.5, .5)
    return xp.clip(batch_action + noise, -1, 1)


class TD3(AttributeSavingMixin, BatchAgent):
    """Twin Delayed Deep Deterministic Policy Gradients (TD3).

    See http://arxiv.org/abs/1802.09477

    Args:
        policy (Policy): Policy.
        q_func1 (Link): First Q-function that takes state-action pairs as input
            and outputs predicted Q-values.
        q_func2 (Link): Second Q-function that takes state-action pairs as
            input and outputs predicted Q-values.
        policy_optimizer (Optimizer): Optimizer setup with the policy
        q_func1_optimizer (Optimizer): Optimizer setup with the first
            Q-function.
        q_func2_optimizer (Optimizer): Optimizer setup with the second
            Q-function.
        replay_buffer (ReplayBuffer): Replay buffer
        gamma (float): Discount factor
        explorer (Explorer): Explorer that specifies an exploration strategy.
        gpu (int): GPU device id if not None nor negative.
        replay_start_size (int): if the replay buffer's size is less than
            replay_start_size, skip update
        minibatch_size (int): Minibatch size
        update_interval (int): Model update interval in step
        phi (callable): Feature extractor applied to observations
        soft_update_tau (float): Tau of soft target update.
        logger (Logger): Logger used
        batch_states (callable): method which makes a batch of observations.
            default is `chainerrl.misc.batch_states.batch_states`
        burnin_action_func (callable or None): If not None, this callable
            object is used to select actions before the model is updated
            one or more times during training.
        policy_update_delay (int): Delay of policy updates. Policy is updated
            once in `policy_update_delay` times of Q-function updates.
        target_policy_smoothing_func (callable): Callable that takes a batch of
            actions as input and outputs a noisy version of it. It is used for
            target policy smoothing when computing target Q-values.
    """

    saved_attributes = (
        'policy',
        'q_func1',
        'q_func2',
        'target_policy',
        'target_q_func1',
        'target_q_func2',
        'policy_optimizer',
        'q_func1_optimizer',
        'q_func2_optimizer',
    )

    def __init__(
            self,
            policy,
            q_func1,
            q_func2,
            policy_optimizer,
            q_func1_optimizer,
            q_func2_optimizer,
            replay_buffer,
            gamma,
            explorer,
            gpu=None,
            replay_start_size=10000,
            minibatch_size=100,
            update_interval=1,
            phi=lambda x: x,
            soft_update_tau=5e-3,
            n_times_update=1,
            logger=getLogger(__name__),
            batch_states=batch_states,
            burnin_action_func=None,
            policy_update_delay=2,
            target_policy_smoothing_func=default_target_policy_smoothing_func,
    ):

        self.policy = policy
        self.q_func1 = q_func1
        self.q_func2 = q_func2

        if gpu is not None and gpu >= 0:
            cuda.get_device_from_id(gpu).use()
            self.policy.to_gpu(device=gpu)
            self.q_func1.to_gpu(device=gpu)
            self.q_func2.to_gpu(device=gpu)

        self.xp = self.policy.xp
        self.replay_buffer = replay_buffer
        self.gamma = gamma
        self.explorer = explorer
        self.gpu = gpu
        self.phi = phi
        self.soft_update_tau = soft_update_tau
        self.logger = logger
        self.policy_optimizer = policy_optimizer
        self.q_func1_optimizer = q_func1_optimizer
        self.q_func2_optimizer = q_func2_optimizer
        self.replay_updater = ReplayUpdater(
            replay_buffer=replay_buffer,
            update_func=self.update,
            batchsize=minibatch_size,
            n_times_update=1,
            replay_start_size=replay_start_size,
            update_interval=update_interval,
            episodic_update=False,
        )
        self.batch_states = batch_states
        self.burnin_action_func = burnin_action_func
        self.policy_update_delay = policy_update_delay
        self.target_policy_smoothing_func = target_policy_smoothing_func

        self.t = 0
        self.last_state = None
        self.last_action = None

        # Target model
        self.target_policy = copy.deepcopy(self.policy)
        self.target_q_func1 = copy.deepcopy(self.q_func1)
        self.target_q_func2 = copy.deepcopy(self.q_func2)

        # Statistics
        self.q1_record = collections.deque(maxlen=1000)
        self.q2_record = collections.deque(maxlen=1000)
        self.q_func1_loss_record = collections.deque(maxlen=100)
        self.q_func2_loss_record = collections.deque(maxlen=100)

    def sync_target_network(self):
        """Synchronize target network with current network."""
        synchronize_parameters(
            src=self.policy,
            dst=self.target_policy,
            method='soft',
            tau=self.soft_update_tau,
        )
        synchronize_parameters(
            src=self.q_func1,
            dst=self.target_q_func1,
            method='soft',
            tau=self.soft_update_tau,
        )
        synchronize_parameters(
            src=self.q_func2,
            dst=self.target_q_func2,
            method='soft',
            tau=self.soft_update_tau,
        )

    def update_q_func(self, batch):
        """Compute loss for a given Q-function."""

        batch_next_state = batch['next_state']
        batch_rewards = batch['reward']
        batch_terminal = batch['is_state_terminal']
        batch_state = batch['state']
        batch_actions = batch['action']
        batch_discount = batch['discount']

        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            next_actions = self.target_policy_smoothing_func(
                self.target_policy(batch_next_state).sample().array)
            next_q1 = self.target_q_func1(batch_next_state, next_actions)
            next_q2 = self.target_q_func2(batch_next_state, next_actions)
            next_q = F.minimum(next_q1, next_q2)

            target_q = batch_rewards + batch_discount * \
                (1.0 - batch_terminal) * F.flatten(next_q)

        predict_q1 = F.flatten(self.q_func1(batch_state, batch_actions))
        predict_q2 = F.flatten(self.q_func2(batch_state, batch_actions))

        loss1 = F.mean_squared_error(target_q, predict_q1)
        loss2 = F.mean_squared_error(target_q, predict_q2)

        # Update stats
        self.q1_record.extend(cuda.to_cpu(predict_q1.array))
        self.q2_record.extend(cuda.to_cpu(predict_q2.array))
        self.q_func1_loss_record.append(float(loss1.array))
        self.q_func2_loss_record.append(float(loss2.array))

        self.q_func1_optimizer.update(lambda: loss1)
        self.q_func2_optimizer.update(lambda: loss2)

    def update_policy(self, batch):
        """Compute loss for actor."""

        batch_state = batch['state']

        onpolicy_actions = self.policy(batch_state).sample()
        q = self.q_func1(batch_state, onpolicy_actions)

        # Since we want to maximize Q, loss is negation of Q
        loss = - F.mean(q)

        self.policy_optimizer.update(lambda: loss)

    def update(self, experiences, errors_out=None):
        """Update the model from experiences"""

        batch = batch_experiences(experiences, self.xp, self.phi, self.gamma)
        self.update_q_func(batch)
        if self.q_func1_optimizer.t % self.policy_update_delay == 0:
            self.update_policy(batch)
            self.sync_target_network()

    def select_onpolicy_action(self, obs):
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            s = self.batch_states([obs], self.xp, self.phi)
            action = self.policy(s).sample().array
        return cuda.to_cpu(action)[0]

    def act_and_train(self, obs, reward):

        self.logger.debug('t:%s r:%s', self.t, reward)

        if (self.burnin_action_func is not None
                and self.policy_optimizer.t == 0):
            action = self.burnin_action_func()
        else:
            onpolicy_action = self.select_onpolicy_action(obs)
            action = self.explorer.select_action(
                self.t, lambda: onpolicy_action)
        self.t += 1

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

        return self.last_action

    def act(self, obs):
        return self.select_onpolicy_action(obs)

    def batch_select_onpolicy_action(self, batch_obs):
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            batch_xs = self.batch_states(batch_obs, self.xp, self.phi)
            batch_action = self.policy(batch_xs).sample().array
        return list(cuda.to_cpu(batch_action))

    def batch_act(self, batch_obs):
        return self.batch_select_onpolicy_action(batch_obs)

    def batch_act_and_train(self, batch_obs):
        """Select a batch of actions for training.

        Args:
            batch_obs (Sequence of ~object): Observations.

        Returns:
            Sequence of ~object: Actions.
        """

        if (self.burnin_action_func is not None
                and self.policy_optimizer.t == 0):
            batch_action = [self.burnin_action_func()
                            for _ in range(len(batch_obs))]
        else:
            batch_onpolicy_action = self.batch_select_onpolicy_action(
                batch_obs)
            batch_action = [
                self.explorer.select_action(
                    self.t, lambda: batch_onpolicy_action[i])
                for i in range(len(batch_onpolicy_action))]

        self.batch_last_obs = list(batch_obs)
        self.batch_last_action = list(batch_action)

        return batch_action

    def batch_observe_and_train(
            self, batch_obs, batch_reward, batch_done, batch_reset):
        for i in range(len(batch_obs)):
            self.t += 1
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

    def batch_observe(self, batch_obs, batch_reward,
                      batch_done, batch_reset):
        pass

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
        self.replay_buffer.stop_current_episode()

    def get_statistics(self):
        return [
            ('average_q1', _mean_or_nan(self.q1_record)),
            ('average_q2', _mean_or_nan(self.q2_record)),
            ('average_q_func1_loss', _mean_or_nan(self.q_func1_loss_record)),
            ('average_q_func2_loss', _mean_or_nan(self.q_func2_loss_record)),
            ('policy_n_updates', self.policy_optimizer.t),
            ('q_func_n_updates', self.q_func1_optimizer.t),
        ]
