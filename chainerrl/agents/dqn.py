from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
import copy
import threading
import os
from logging import getLogger
from concurrent.futures import ThreadPoolExecutor
from future.utils import native

import numpy as np
import chainer
import chainer.functions as F
from chainer import cuda
from chainer import serializers

from chainerrl import agent
from chainerrl.misc.makedirs import makedirs
from chainerrl.misc import copy_param
from chainerrl.recurrent import Recurrent
from chainerrl.recurrent import state_reset


def _to_device(obj, gpu):
    if isinstance(obj, tuple):
        return tuple(_to_device(x, gpu) for x in obj)
    else:
        if gpu >= 0:
            return cuda.to_gpu(obj, gpu)
        else:
            return cuda.to_cpu(obj)


def batch_experiences(experiences, xp, phi):
    return {
        'state': xp.asarray([phi(elem['state']) for elem in experiences]),
        'action': xp.asarray([elem['action'] for elem in experiences]),
        'reward': xp.asarray(
            [elem['reward'] for elem in experiences], dtype=np.float32),
        'next_state': xp.asarray(
            [phi(elem['next_state']) for elem in experiences]),
        'next_action': xp.asarray(
            [elem['next_action'] for elem in experiences]),
        'is_state_terminal': xp.asarray(
            [elem['is_state_terminal'] for elem in experiences],
            dtype=np.float32)}


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


class DQN(agent.Agent):
    """Deep Q-Network algorithm.

    Args:
        q_function (StateQFunction): Q-function
        optimizer (Optimizer): Optimizer that is already setup
        replay_buffer (ReplayBuffer): Replay buffer
        gamma (float): Discount factor
        explorer (Explorer): Explorer that specifies an exploration strategy.
        gpu (int): GPU device id. -1 for CPU.
        replay_start_size (int): if the replay buffer's size is less than
            replay_start_size, skip update
        minibatch_size (int): Minibatch size
        update_frequency (int): Model update frequency in step
        target_update_frequency (int): Target model update frequency in step
        clip_delta (bool): Clip delta if set True
        phi (callable): Feature extractor applied to observations
        target_update_method (str): 'hard' or 'soft'.
        soft_update_tau (float): Tau of soft target update.
        async_update (bool): Update model in a different thread if set True
        n_times_update (int): Number of repetition of update
        average_q_decay (float): Decay rate of average Q, only used for
            statistics
        average_loss_decay (float): Decay rate of average loss, only used for
            statistics
        batch_accumulator (str): 'mean' or 'sum'
        episodic_update (bool): Use full episodes for update if set True
        episodic_update_len (int or None): Subsequences of this length are used
            for update if set int and episodic_update=True
        logger (Logger): Logger used
    """

    def __init__(self, q_function, optimizer, replay_buffer, gamma,
                 explorer, gpu=-1, replay_start_size=50000,
                 minibatch_size=32, update_frequency=1,
                 target_update_frequency=10000, clip_delta=True,
                 clip_reward=True, phi=lambda x: x,
                 target_update_method='hard',
                 soft_update_tau=1e-2, async_update=False,
                 n_times_update=1, average_q_decay=0.999,
                 average_loss_decay=0.99,
                 batch_accumulator='mean', episodic_update=False,
                 episodic_update_len=None,
                 logger=getLogger(__name__)):
        self.model = q_function
        self.q_function = q_function  # For backward compatibility

        # Future's builtins.int is a new type that inherit long, but Chainer
        # 1.15 only accepts int and long, so here we should use a native type.
        gpu = native(gpu)

        if gpu >= 0:
            cuda.get_device(gpu).use()
            self.model.to_gpu(device=gpu)
            self.xp = cuda.cupy
        else:
            self.xp = np
        self.replay_buffer = replay_buffer
        self.optimizer = optimizer
        self.gamma = gamma
        self.explorer = explorer
        self.gpu = gpu
        assert minibatch_size <= replay_start_size
        self.replay_start_size = replay_start_size
        self.minibatch_size = minibatch_size
        self.update_frequency = update_frequency
        self.target_update_frequency = target_update_frequency
        self.clip_delta = clip_delta
        self.clip_reward = clip_reward
        self.phi = phi
        self.target_update_method = target_update_method
        self.soft_update_tau = soft_update_tau
        self.async_update = async_update
        self.n_times_update = n_times_update
        self.batch_accumulator = batch_accumulator
        assert batch_accumulator in ('mean', 'sum')
        self.logger = logger

        self.t = 0
        self.last_state = None
        self.last_action = None
        self.lock = threading.Lock()
        if self.async_update:
            self.update_executor = ThreadPoolExecutor(max_workers=1)
            self.update_executor.submit(
                lambda: cuda.get_device(gpu).use()).result()
            self.update_future = None
        self.episodic_update = episodic_update
        self.episodic_update_len = episodic_update_len
        self.target_model = None
        self.sync_target_network()
        self.target_q_function = self.target_model  # For backward compatibility
        self.average_q = 0
        self.average_q_decay = average_q_decay
        self.average_loss = 0
        self.average_loss_decay = average_loss_decay

    def sync_target_network(self):
        """Synchronize target network with current network."""
        if self.target_model is None:
            self.target_model = copy.deepcopy(self.model)
        else:
            if self.target_update_method == 'hard':
                self.logger.debug('sync')
                copy_param.copy_param(self.target_model, self.model)
            elif self.target_update_method == 'soft':
                copy_param.soft_copy_param(
                    self.target_model, self.model, self.soft_update_tau)
            else:
                raise RuntimeError('Unknown target update method: {}'.format(
                    self.target_update_method))

    def update(self, experiences, errors_out=None):
        """Update the model from experiences

        This function is thread-safe.
        Args:
          experiences (list): list of dict that contains
            state: cupy.ndarray or numpy.ndarray
            action: int [0, n_action_types)
            reward: float32
            next_state: cupy.ndarray or numpy.ndarray
            next_legal_actions: list of booleans; True means legal
          gamma (float): discount factor
        Returns:
          None
        """

        exp_batch = batch_experiences(experiences, xp=self.xp, phi=self.phi)
        loss = self._compute_loss(
            exp_batch, self.gamma, errors_out=errors_out)

        # Update stats
        self.average_loss *= self.average_loss_decay
        self.average_loss += (1 - self.average_loss_decay) * float(loss.data)

        self.optimizer.zero_grads()
        loss.backward()
        self.optimizer.update()

    def input_initial_batch_to_target_model(self, batch):
        self.target_model(batch['state'])

    def update_from_episodes(self, episodes, errors_out=None):
        with state_reset(self.model):
            with state_reset(self.target_model):
                loss = 0
                sorted_episodes = list(reversed(sorted(episodes, key=len)))
                max_epi_len = len(sorted_episodes[0])
                for i in range(max_epi_len):
                    transitions = []
                    for ep in sorted_episodes:
                        if len(ep) <= i:
                            break
                        transitions.append(ep[i])
                    batch = batch_experiences(transitions,
                                              xp=self.xp, phi=self.phi)
                    if i == 0:
                        self.input_initial_batch_to_target_model(batch)
                    loss += self._compute_loss(batch, self.gamma)
                loss /= max_epi_len
                self.optimizer.zero_grads()
                loss.backward()
                self.optimizer.update()

    def _batch_states(self, states):
        """Generate an input batch array from a list of states
        """
        states = [self.phi(s) for s in states]
        return self.xp.asarray(states)

    def _compute_target_values(self, exp_batch, gamma):

        batch_next_state = exp_batch['next_state']

        target_next_qout = self.target_model(batch_next_state, test=True)
        next_q_max = target_next_qout.max

        batch_rewards = exp_batch['reward']
        batch_terminal = exp_batch['is_state_terminal']

        return batch_rewards + self.gamma * (1.0 - batch_terminal) * next_q_max

    def _compute_y_and_t(self, exp_batch, gamma):

        batch_size = exp_batch['state'].shape[0]

        # Compute Q-values for current states
        batch_state = exp_batch['state']

        qout = self.model(batch_state, test=False)

        batch_actions = exp_batch['action']
        batch_q = F.reshape(qout.evaluate_actions(
            batch_actions), (batch_size, 1))

        with chainer.no_backprop_mode():
            batch_q_target = F.reshape(
                self._compute_target_values(exp_batch, gamma),
                (batch_size, 1))

        return batch_q, batch_q_target

    def _compute_loss(self, exp_batch, gamma, errors_out=None):
        """
        Compute the Q-learning loss for a batch of experiences


        Args:
          experiences (list): see update()'s docstring
          gamma (float): discount factor
        Returns:
          loss
        """

        y, t = self._compute_y_and_t(exp_batch, gamma)

        if errors_out is not None:
            del errors_out[:]
            delta = F.sum(F.basic_math.absolute(y - t), axis=1)
            delta = cuda.to_cpu(delta.data)
            for e in delta:
                errors_out.append(e)

        return compute_value_loss(y, t, clip_delta=self.clip_delta,
                                  batch_accumulator=self.batch_accumulator)

    def compute_q_values(self, states):
        """Compute Q-values

        This function is thread-safe.
        Args:
          states (list of cupy.ndarray or numpy.ndarray)
        Returns:
          list of numpy.ndarray
        """
        if not states:
            return []
        self.lock.acquire()
        batch_x = self._batch_states(states)
        q_values = list(cuda.to_cpu(
            self.model(batch_x, test=True).q_values))
        self.lock.release()
        return q_values

    def _to_my_device(self, model):
        if self.gpu >= 0:
            model.to_gpu(self.gpu)
        else:
            model.to_cpu()

    @property
    def saved_attributes(self):
        return ('model', 'target_model', 'optimizer')

    def save(self, dirname):
        makedirs(dirname, exist_ok=True)
        for attr in self.saved_attributes:
            serializers.save_npz(
                os.path.join(dirname, '{}.npz'.format(attr)),
                getattr(self, attr))

    def load(self, dirname):
        for attr in self.saved_attributes:
            serializers.load_npz(
                os.path.join(dirname, '{}.npz'.format(attr)),
                getattr(self, attr))

    def act(self, state):
        qout = self.model(self._batch_states([state]), test=True)
        action = cuda.to_cpu(qout.greedy_actions.data)[0]
        action_var = chainer.Variable(self.xp.asarray([action]))
        q = float(qout.evaluate_actions(action_var).data)

        # Update stats
        self.average_q *= self.average_q_decay
        self.average_q += (1 - self.average_q_decay) * q

        self.logger.debug('t:%s a:%s q:%s qout:%s', self.t, action, q, qout)
        return action

    def update_if_necessary(self):
        if self.t < self.replay_start_size:
            return
        if self.t % self.update_frequency != 0:
            return

        def update_func():
            for _ in range(self.n_times_update):
                if self.episodic_update:
                    episodes = self.replay_buffer.sample_episodes(
                        self.minibatch_size, self.episodic_update_len)
                    self.update_from_episodes(episodes)
                else:
                    experiences = self.replay_buffer.sample(
                        self.minibatch_size)
                    self.update(experiences)

        if self.async_update:
            self.update_future = self.update_executor.submit(update_func)
        else:
            update_func()

    def act_and_train(self, state, reward):

        self.logger.debug('t:%s r:%s', self.t, reward)

        if self.async_update and self.update_future:
            self.update_future.result()
            self.update_future = None

        if self.clip_reward:
            reward = np.clip(reward, -1, 1)

        greedy_action = self.act(state)
        action = self.explorer.select_action(self.t, lambda: greedy_action)
        self.t += 1

        # Update the target network
        if self.t % self.target_update_frequency == 0:
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

        self.update_if_necessary()

        return self.last_action

    def stop_episode_and_train(self, state, reward, done=False):
        """
        Observe a terminal state and a reward.

        This function must be called once when an episode terminates.
        """

        if self.clip_reward:
            reward = np.clip(reward, -1, 1)

        assert self.last_state is not None
        assert self.last_action is not None

        if self.async_update and self.update_future:
            self.update_future.result()
            self.update_future = None

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

    def get_stats_keys(self):
        return ('average_q', 'average_loss')

    def get_stats_values(self):
        return (self.average_q, self.average_loss)
