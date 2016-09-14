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
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from future.utils import native

import numpy as np
import chainer
import chainer.functions as F
from chainer import cuda
from chainer import serializers

import agent
import copy_param


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
            [[elem['reward']] for elem in experiences], dtype=np.float32),
        'next_state': xp.asarray(
            [phi(elem['next_state']) for elem in experiences]),
        'is_state_terminal': xp.asarray(
            [[elem['is_state_terminal']] for elem in experiences],
            dtype=np.float32)}


class DQN(agent.Agent):
    """DQN = QNetwork + Optimizer
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
                 batch_accumulator='mean',
                 logger=getLogger(__name__)):
        """
        Args:
          replay_start_size (int): if replay buffer's size is less than
            replay_start_size, skip updating
          target_update_frequency (int): frequency of updating target Q
            function
          target_update_method (str): 'hard' or 'soft'.
          soft_update_tau (float): tau of soft target update.
        """
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
        loss = self._compute_loss(
            experiences, self.gamma, errors_out=errors_out)

        # Update stats
        self.average_loss *= self.average_loss_decay
        self.average_loss += (1 - self.average_loss_decay) * float(loss.data)

        self.optimizer.zero_grads()
        loss.backward()
        self.optimizer.update()

    def _batch_states(self, states):
        """Generate an input batch array from a list of states
        """
        states = [self.phi(s) for s in states]
        return self.xp.asarray(states)

    def _compute_target_values(self, experiences, gamma):

        batch_next_state = self._batch_states(
            [elem['next_state'] for elem in experiences])

        target_next_qout = self.target_model(batch_next_state, test=True)
        next_q_max = target_next_qout.max
        next_q_max.creator = None

        batch_rewards = self.xp.asarray(
            [elem['reward'] for elem in experiences], dtype=np.float32)

        batch_terminal = self.xp.asarray(
            [elem['is_state_terminal'] for elem in experiences],
            dtype=np.float32)

        return batch_rewards + self.gamma * (1.0 - batch_terminal) * next_q_max

    def _compute_y_and_t(self, experiences, gamma):

        batch_size = len(experiences)

        # Compute Q-values for current states
        batch_state = self._batch_states(
            [elem['state'] for elem in experiences])

        qout = self.model(batch_state, test=False)

        batch_actions = self.xp.asarray(
            [elem['action'] for elem in experiences])
        batch_q = F.reshape(qout.evaluate_actions(
            batch_actions), (batch_size, 1))

        batch_q_target = F.reshape(
            self._compute_target_values(experiences, gamma), (batch_size, 1))

        batch_q_target.creator = None

        return batch_q, batch_q_target

    def _compute_loss(self, experiences, gamma, errors_out=None):
        """
        Compute the Q-learning loss for a batch of experiences


        Args:
          experiences (list): see update()'s docstring
          gamma (float): discount factor
        Returns:
          loss
        """
        assert experiences

        y, t = self._compute_y_and_t(experiences, gamma)

        if errors_out is not None:
            del errors_out[:]
            delta = F.sum(F.basic_math.absolute(y - t), axis=1)
            delta = cuda.to_cpu(delta.data)
            for e in delta:
                errors_out.append(e)

        if self.clip_delta:
            loss_sum = F.sum(F.huber_loss(y, t, delta=1.0))
            if self.batch_accumulator == 'mean':
                loss = loss_sum / len(experiences)
            elif self.batch_accumulator == 'sum':
                loss = loss_sum
        else:
            loss_mean = F.mean_squared_error(y, t) / 2
            if self.batch_accumulator == 'mean':
                loss = loss_mean
            elif self.batch_accumulator == 'sum':
                loss = loss_mean * len(experiences)
        return loss

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

    def _load_model_without_lock(self, model_filename):
        serializers.load_hdf5(model_filename, self.model)

        # Load target model
        target_filename = model_filename + '.target'
        if os.path.exists(target_filename):
            serializers.load_hdf5(target_filename, self.target_model)
        else:
            print('WARNING: {0} was not found'.format(target_filename))
            copy_param.copy_param(target_link=self.target_model,
                                  source_link=self.model)

        self.sync_target_network()
        opt_filename = model_filename + '.opt'
        if os.path.exists(opt_filename):
            serializers.load_hdf5(model_filename + '.opt', self.optimizer)
        else:
            print('WARNING: {0} was not found, so loaded only a model'.format(
                opt_filename))

    def load_model(self, model_filename):
        """Load a network model form a file

        This function is thread-safe.
        """
        self.lock.acquire()
        self._load_model_without_lock(model_filename)
        self.lock.release()

    def save_model(self, model_filename):
        """Save a network model to a file

        This function is thread-safe.
        """
        self.lock.acquire()
        serializers.save_hdf5(model_filename, self.model)
        serializers.save_hdf5(model_filename + '.target', self.target_model)
        serializers.save_hdf5(model_filename + '.opt', self.optimizer)
        self.lock.release()

    def select_greedy_action(self, state):
        qout = self.model(self._batch_states([state]), test=True)
        action = cuda.to_cpu(qout.greedy_actions.data)[0]
        action_var = chainer.Variable(self.xp.asarray([action]))
        q = float(qout.evaluate_actions(action_var).data)

        # Update stats
        self.average_q *= self.average_q_decay
        self.average_q += (1 - self.average_q_decay) * q

        self.logger.debug('t:%s a:%s q:%s qout:%s', self.t, action, q, qout)
        return action

    def select_action(self, state):
        return self.explorer.select_action(
            self.t, lambda: self.select_greedy_action(state))

    def act(self, state, reward):
        """
        Observe a current state and a reward, then choose an action.

        This function must be called every time step exept at terminal states.
        """

        self.logger.debug('t:%s r:%s', self.t, reward)

        if self.async_update and self.update_future:
            self.update_future.result()
            self.update_future = None

        if self.clip_reward:
            reward = np.clip(reward, -1, 1)

        action = self.select_action(state)
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
                is_state_terminal=False)

        self.last_state = state
        self.last_action = action

        if len(self.replay_buffer) >= self.replay_start_size and \
                self.t % self.update_frequency == 0:
            def update_func():
                for _ in range(self.n_times_update):
                    experiences = self.replay_buffer.sample(
                        self.minibatch_size)
                    self.update(experiences)
            if self.async_update:
                self.update_future = self.update_executor.submit(update_func)
            else:
                update_func()

        return self.last_action

    def observe_terminal(self, state, reward):
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
            is_state_terminal=True)

        self.last_state = None
        self.last_action = None

    def stop_current_episode(self):
        """
        Stop the current episode.

        This function must be called once when an episode is stopped.
        """
        self.last_state = None
        self.last_action = None

    def get_stats_keys(self):
        return ('average_q', 'average_loss')

    def get_stats_values(self):
        return (self.average_q, self.average_loss)
