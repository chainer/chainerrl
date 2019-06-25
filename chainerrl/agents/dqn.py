from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import copy
from logging import getLogger

import chainer
from chainer import cuda
import chainer.functions as F

from chainerrl import agent
from chainerrl.misc.batch_states import batch_states
from chainerrl.misc.copy_param import synchronize_parameters
from chainerrl.recurrent import Recurrent
from chainerrl.recurrent import state_reset
from chainerrl.replay_buffer import batch_experiences
from chainerrl.replay_buffer import ReplayUpdater


def compute_value_loss(y, t, clip_delta=True, batch_accumulator='mean'):
    """Compute a loss for value prediction problem.

    Args:
        y (Variable or ndarray): Predicted values.
        t (Variable or ndarray): Target values.
        clip_delta (bool): Use the Huber loss function if set True.
        batch_accumulator (str): 'mean' or 'sum'. 'mean' will use the mean of
            the loss values in a batch. 'sum' will use the sum.
    Returns:
        (Variable) scalar loss
    """
    assert batch_accumulator in ('mean', 'sum')
    y = F.reshape(y, (-1, 1))
    t = F.reshape(t, (-1, 1))
    if clip_delta:
        loss_sum = F.sum(F.huber_loss(y, t, delta=1.0))
        if batch_accumulator == 'mean':
            loss = loss_sum / y.shape[0]
        elif batch_accumulator == 'sum':
            loss = loss_sum
    else:
        loss_mean = F.mean_squared_error(y, t) / 2
        if batch_accumulator == 'mean':
            loss = loss_mean
        elif batch_accumulator == 'sum':
            loss = loss_mean * y.shape[0]
    return loss


def compute_weighted_value_loss(y, t, weights,
                                clip_delta=True, batch_accumulator='mean'):
    """Compute a loss for value prediction problem.

    Args:
        y (Variable or ndarray): Predicted values.
        t (Variable or ndarray): Target values.
        weights (ndarray): Weights for y, t.
        clip_delta (bool): Use the Huber loss function if set True.
        batch_accumulator (str): 'mean' will divide loss by batchsize
    Returns:
        (Variable) scalar loss
    """
    assert batch_accumulator in ('mean', 'sum')
    y = F.reshape(y, (-1, 1))
    t = F.reshape(t, (-1, 1))
    if clip_delta:
        losses = F.huber_loss(y, t, delta=1.0)
    else:
        losses = F.square(y - t) / 2
    losses = F.reshape(losses, (-1,))
    loss_sum = F.sum(losses * weights)
    if batch_accumulator == 'mean':
        loss = loss_sum / y.shape[0]
    elif batch_accumulator == 'sum':
        loss = loss_sum
    return loss


class DQN(agent.AttributeSavingMixin, agent.BatchAgent):
    """Deep Q-Network algorithm.

    Args:
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
    """

    saved_attributes = ('model', 'target_model', 'optimizer')

    def __init__(self, q_function, optimizer, replay_buffer, gamma,
                 explorer, gpu=None, replay_start_size=50000,
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
        self.model = q_function
        self.q_function = q_function  # For backward compatibility

        if gpu is not None and gpu >= 0:
            cuda.get_device(gpu).use()
            self.model.to_gpu(device=gpu)

        self.xp = self.model.xp
        self.replay_buffer = replay_buffer
        self.optimizer = optimizer
        self.gamma = gamma
        self.explorer = explorer
        self.gpu = gpu
        self.target_update_interval = target_update_interval
        self.clip_delta = clip_delta
        self.phi = phi
        self.target_update_method = target_update_method
        self.soft_update_tau = soft_update_tau
        self.batch_accumulator = batch_accumulator
        assert batch_accumulator in ('mean', 'sum')
        self.logger = logger
        self.batch_states = batch_states
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

        self.t = 0
        self.last_state = None
        self.last_action = None
        self.target_model = None
        self.sync_target_network()
        # For backward compatibility
        self.target_q_function = self.target_model
        self.average_q = 0
        self.average_q_decay = average_q_decay
        self.average_loss = 0
        self.average_loss_decay = average_loss_decay

        # Error checking
        if (self.replay_buffer.capacity is not None and
                self.replay_buffer.capacity <
                self.replay_updater.replay_start_size):
            raise ValueError(
                'Replay start size cannot exceed '
                'replay buffer capacity.')

    def sync_target_network(self):
        """Synchronize target network with current network."""
        if self.target_model is None:
            self.target_model = copy.deepcopy(self.model)
            call_orig = self.target_model.__call__

            def call_test(self_, x):
                with chainer.using_config('train', False):
                    return call_orig(self_, x)

            self.target_model.__call__ = call_test
        else:
            synchronize_parameters(
                src=self.model,
                dst=self.target_model,
                method=self.target_update_method,
                tau=self.soft_update_tau)

    def update(self, experiences, errors_out=None):
        """Update the model from experiences

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
        has_weight = 'weight' in experiences[0][0]
        exp_batch = batch_experiences(
            experiences, xp=self.xp,
            phi=self.phi, gamma=self.gamma,
            batch_states=self.batch_states)
        if has_weight:
            exp_batch['weights'] = self.xp.asarray(
                [elem[0]['weight']for elem in experiences],
                dtype=self.xp.float32)
            if errors_out is None:
                errors_out = []
        loss = self._compute_loss(exp_batch, errors_out=errors_out)
        if has_weight:
            self.replay_buffer.update_errors(errors_out)

        # Update stats
        self.average_loss *= self.average_loss_decay
        self.average_loss += (1 - self.average_loss_decay) * float(loss.array)

        self.model.cleargrads()
        loss.backward()
        self.optimizer.update()

    def input_initial_batch_to_target_model(self, batch):
        self.target_model(batch['state'])

    def update_from_episodes(self, episodes, errors_out=None):
        has_weights = isinstance(episodes, tuple)
        if has_weights:
            episodes, weights = episodes
            if errors_out is None:
                errors_out = []
        if errors_out is None:
            errors_out_step = None
        else:
            del errors_out[:]
            for _ in episodes:
                errors_out.append(0.0)
            errors_out_step = []

        with state_reset(self.model), state_reset(self.target_model):
            loss = 0
            tmp = list(reversed(sorted(
                enumerate(episodes), key=lambda x: len(x[1]))))
            sorted_episodes = [elem[1] for elem in tmp]
            indices = [elem[0] for elem in tmp]  # argsort
            max_epi_len = len(sorted_episodes[0])
            for i in range(max_epi_len):
                transitions = []
                weights_step = []
                for ep, index in zip(sorted_episodes, indices):
                    if len(ep) <= i:
                        break
                    transitions.append([ep[i]])
                    if has_weights:
                        weights_step.append(weights[index])
                batch = batch_experiences(
                    transitions,
                    xp=self.xp,
                    phi=self.phi,
                    gamma=self.gamma,
                    batch_states=self.batch_states)
                assert len(batch['state']) == len(transitions)
                if i == 0:
                    self.input_initial_batch_to_target_model(batch)
                if has_weights:
                    batch['weights'] = self.xp.asarray(
                        weights_step, dtype=self.xp.float32)
                loss += self._compute_loss(batch,
                                           errors_out=errors_out_step)
                if errors_out is not None:
                    for err, index in zip(errors_out_step, indices):
                        errors_out[index] += err
            loss /= max_epi_len

            # Update stats
            self.average_loss *= self.average_loss_decay
            self.average_loss += \
                (1 - self.average_loss_decay) * float(loss.array)

            self.model.cleargrads()
            loss.backward()
            self.optimizer.update()
        if has_weights:
            self.replay_buffer.update_errors(errors_out)

    def _compute_target_values(self, exp_batch):
        batch_next_state = exp_batch['next_state']

        target_next_qout = self.target_model(batch_next_state)
        next_q_max = target_next_qout.max

        batch_rewards = exp_batch['reward']
        batch_terminal = exp_batch['is_state_terminal']
        discount = exp_batch['discount']

        return batch_rewards + discount * (1.0 - batch_terminal) * next_q_max

    def _compute_y_and_t(self, exp_batch):
        batch_size = exp_batch['reward'].shape[0]

        # Compute Q-values for current states
        batch_state = exp_batch['state']

        qout = self.model(batch_state)

        batch_actions = exp_batch['action']
        batch_q = F.reshape(qout.evaluate_actions(
            batch_actions), (batch_size, 1))

        with chainer.no_backprop_mode():
            batch_q_target = F.reshape(
                self._compute_target_values(exp_batch),
                (batch_size, 1))

        return batch_q, batch_q_target

    def _compute_loss(self, exp_batch, errors_out=None):
        """Compute the Q-learning loss for a batch of experiences


        Args:
          exp_batch (dict): A dict of batched arrays of transitions
        Returns:
          Computed loss from the minibatch of experiences
        """
        y, t = self._compute_y_and_t(exp_batch)

        if errors_out is not None:
            del errors_out[:]
            delta = F.absolute(y - t)
            if delta.ndim == 2:
                delta = F.sum(delta, axis=1)
            delta = cuda.to_cpu(delta.array)
            for e in delta:
                errors_out.append(e)

        if 'weights' in exp_batch:
            return compute_weighted_value_loss(
                y, t, exp_batch['weights'],
                clip_delta=self.clip_delta,
                batch_accumulator=self.batch_accumulator)
        else:
            return compute_value_loss(y, t, clip_delta=self.clip_delta,
                                      batch_accumulator=self.batch_accumulator)

    def act(self, obs):
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            action_value = self.model(
                self.batch_states([obs], self.xp, self.phi))
            q = float(action_value.max.array)
            action = cuda.to_cpu(action_value.greedy_actions.array)[0]

        # Update stats
        self.average_q *= self.average_q_decay
        self.average_q += (1 - self.average_q_decay) * q

        self.logger.debug('t:%s q:%s action_value:%s', self.t, q, action_value)
        return action

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

    def batch_act_and_train(self, batch_obs):
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            batch_xs = self.batch_states(batch_obs, self.xp, self.phi)
            batch_av = self.model(batch_xs)
            batch_maxq = batch_av.max.array
            batch_argmax = cuda.to_cpu(batch_av.greedy_actions.array)
        batch_action = [
            self.explorer.select_action(
                self.t, lambda: batch_argmax[i],
                action_value=batch_av[i:i + 1],
            )
            for i in range(len(batch_obs))]
        self.batch_last_obs = list(batch_obs)
        self.batch_last_action = list(batch_action)

        # Update stats
        self.average_q *= self.average_q_decay
        self.average_q += (1 - self.average_q_decay) * float(batch_maxq.mean())

        return batch_action

    def batch_act(self, batch_obs):
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            batch_xs = self.batch_states(batch_obs, self.xp, self.phi)
            batch_av = self.model(batch_xs)
            batch_argmax = cuda.to_cpu(batch_av.greedy_actions.array)
            return batch_argmax

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

    def batch_observe(self, batch_obs, batch_reward,
                      batch_done, batch_reset):
        pass

    def stop_episode_and_train(self, state, reward, done=False):
        """Observe a terminal state and a reward.

        This function must be called once when an episode terminates.
        """

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
            ('average_loss', self.average_loss),
            ('n_updates', self.optimizer.t),
        ]
