import copy
import threading
import unittest
import os
import tempfile

import numpy as np
import chainer.functions as F
from chainer import optimizers, cuda, Variable
from chainer.testing import attr
from chainer import serializers

import agent
import q_function
import copy_param
import smooth_l1_loss


def _to_device(obj, gpu):
    if isinstance(obj, tuple):
        return tuple(_to_device(x, gpu) for x in obj)
    else:
        if gpu >= 0:
            return cuda.to_gpu(obj, gpu)
        else:
            return cuda.to_cpu(obj)


class DQN(agent.Agent):
    """DQN = QNetwork + Optimizer
    """

    def __init__(self, q_function, optimizer, replay_buffer, gamma, epsilon,
                 gpu=-1, replay_start_size=50000, minibatch_size=32,
                 update_frequency=4, target_update_frequency=10000,
                 clip_delta=True):
        """
        Args:
          replay_start_size (int): if replay buffer's size is less than replay_start_size, skip updating
          target_update_frequency (int): frequency of updating target Q function
        """
        self.q_function = q_function
        if gpu >= 0:
            cuda.get_device(gpu).use()
            self.q_function.to_gpu(device=gpu)
        self.target_q_function = copy.deepcopy(self.q_function)
        self.replay_buffer = replay_buffer
        self.optimizer = optimizer
        self.gamma = gamma
        self.epsilon = epsilon
        self.gpu = gpu
        assert minibatch_size <= replay_start_size
        self.replay_start_size = replay_start_size
        self.minibatch_size = minibatch_size
        self.update_frequency = update_frequency
        self.target_update_frequency = target_update_frequency
        self.clip_delta = clip_delta

        self.t = 0
        self.last_state = None
        self.last_action = None
        self.lock = threading.Lock()

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
        self.optimizer.zero_grads()
        loss.backward()
        self.optimizer.update()

    def _batch_states(self, states):
        """Generate an input batch array from a list of states
        """
        if not states:
            return []
        xp = cuda.get_array_module(states[0])
        batch = xp.vstack([state for state in states])
        if self.gpu >= 0:
            batch = cuda.to_gpu(batch, device=self.gpu)
        else:
            batch = cuda.to_cpu(batch)
        return batch

    def _compute_target_values(self, experiences, gamma, batch_q):

        batch_next_state = self._batch_states(
            [elem['next_state'] for elem in experiences])

        batch_next_q = cuda.to_cpu(
            self.target_q_function.forward(batch_next_state, test=True).data)

        batch_q_target = batch_q.copy()

        # Set target values for max actions
        for batch_idx in xrange(len(experiences)):
            experience = experiences[batch_idx]
            action = experience['action']
            reward = experience['reward']
            max_q = batch_next_q[batch_idx].max()
            if experience['is_state_terminal']:
                q_target = reward
            else:
                q_target = reward + self.gamma * max_q
            batch_q_target[batch_idx, action] = q_target

        return batch_q_target

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

        # Compute Q-values for current states
        batch_state = self._batch_states(
            [elem['state'] for elem in experiences])
        batch_q = self.q_function.forward(batch_state, test=False)

        batch_q_target = self._compute_target_values(
            experiences, gamma, cuda.to_cpu(batch_q.data))

        if self.gpu >= 0:
            batch_q_target = cuda.to_gpu(batch_q_target, self.gpu)

        batch_q_target = Variable(batch_q_target)

        if errors_out is not None:
            del errors_out[:]
            delta = F.sum(
                F.basic_math.absolute(batch_q - batch_q_target), axis=1)
            delta = cuda.to_cpu(delta.data)
            for e in delta:
                errors_out.append(e)

        if self.clip_delta:
            return smooth_l1_loss.smooth_l1_loss(batch_q, batch_q_target)
        else:
            return F.sum((batch_q - batch_q_target) ** 2)

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
            self.q_function.forward(batch_x, test=True).data))
        self.lock.release()
        return q_values

    def _to_my_device(self, model):
        if self.gpu >= 0:
            model.to_gpu(self.gpu)
        else:
            model.to_cpu()

    def _load_model_without_lock(self, model_filename):
        serializers.load_hdf5(model_filename, self.q_function)
        copy_param.copy_param(self.target_q_function, self.q_function)
        opt_filename = model_filename + '.opt'
        if os.path.exists(opt_filename):
            print 'WARNING: {0} was not found, so loaded only a model'.format(
                opt_filename)
            serializers.load_hdf5(model_filename + '.opt', self.optimizer)

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
        serializers.save_hdf5(model_filename, self.q_function)
        serializers.save_hdf5(model_filename + '.opt', self.optimizer)
        self.lock.release()

    def act(self, state, reward, is_state_terminal):

        state = state.reshape((1,) + state.shape)

        ret_action = None
        if not is_state_terminal:
            action, q = self.q_function.sample_epsilon_greedily_with_value(
                self._batch_states([state]), self.epsilon)
            action = int(action[0])
            if self.t % 100 == 0:
                print 't:{} q:{}'.format(self.t, q.data)
            self.t += 1

            # Update the target network
            # Global counter T is used in the original paper, but here we use
            # process specific counter instead. So i_target should be set
            # x-times smaller, where x is the number of processes
            if self.t % self.target_update_frequency == 0:
                print 'sync'
                copy_param.copy_param(self.target_q_function, self.q_function)

        if self.last_state is not None:
            assert self.last_action is not None
            # Add a transition to the replay buffer
            self.replay_buffer.append(
                state=self.last_state,
                action=self.last_action,
                reward=reward,
                next_state=state,
                is_state_terminal=is_state_terminal)

        if not is_state_terminal:
            self.last_state = state
            self.last_action = action
        else:
            self.last_state = None
            self.last_action = None

        if len(self.replay_buffer) >= self.replay_start_size and \
                self.t % self.update_frequency == 0:
            experiences = self.replay_buffer.sample(self.minibatch_size)
            self.update(experiences)

        return self.last_action
