import copy
import threading
import os
from logging import getLogger

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


class DQN(agent.Agent):
    """DQN = QNetwork + Optimizer
    """

    def __init__(self, q_function, optimizer, replay_buffer, gamma,
                 explorer, gpu=-1, replay_start_size=50000,
                 minibatch_size=32, update_frequency=1,
                 target_update_frequency=10000, clip_delta=True,
                 clip_reward=True, phi=lambda x: x,
                 logger=getLogger(__name__)):
        """
        Args:
          replay_start_size (int): if replay buffer's size is less than
            replay_start_size, skip updating
          target_update_frequency (int): frequency of updating target Q
            function
        """
        self.q_function = q_function
        if gpu >= 0:
            cuda.get_device(gpu).use()
            self.q_function.to_gpu(device=gpu)
            self.xp = cuda.cupy
        else:
            self.xp = np
        self.target_q_function = copy.deepcopy(self.q_function)
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
        self.logger = logger

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
        states = [self.phi(s) for s in states]
        xp = cuda.get_array_module(states[0])
        batch = xp.asarray(states)
        if self.gpu >= 0:
            batch = cuda.to_gpu(batch, device=self.gpu)
        else:
            batch = cuda.to_cpu(batch)
        return chainer.Variable(batch)

    def _compute_target_values(self, experiences, gamma):

        batch_next_state = self._batch_states(
            [elem['next_state'] for elem in experiences])

        target_next_qout = self.target_q_function(batch_next_state, test=True)
        next_q_max = target_next_qout.max
        next_q_max.unchain_backward()

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

        qout = self.q_function(batch_state, test=False)
        xp = cuda.get_array_module(qout.greedy_actions.data)

        batch_actions = chainer.Variable(
            xp.asarray([elem['action'] for elem in experiences]))
        batch_q = F.reshape(qout.evaluate_actions(
            batch_actions), (batch_size, 1))

        batch_q_target = F.reshape(
            self._compute_target_values(experiences, gamma), (batch_size, 1))

        batch_q_target.unchain_backward()

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
            return F.sum(F.huber_loss(y, t, delta=1.0)) / len(experiences)
        else:
            return F.mean_squared_error(y, t) / 2

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
            self.q_function(batch_x, test=True).q_values))
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
            print('WARNING: {0} was not found, so loaded only a model'.format(
                opt_filename))
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

    def select_action(self, state):
        qout = self.q_function(self._batch_states([state]), test=True)

        action = self.explorer.select_action(
            self.t, lambda: cuda.to_cpu(qout.greedy_actions.data)[0])
        action_var = chainer.Variable(self.xp.asarray([action]))
        q = qout.evaluate_actions(action_var)

        self.logger.debug('t:%s a:%s q:%s qout:%s',
                          self.t, action, q.data, qout)

        return action

    def act(self, state, reward, is_state_terminal):

        self.logger.debug('t:%s r:%s', self.t, reward)

        if self.clip_reward:
            reward = np.clip(reward, -1, 1)

        if not is_state_terminal:
            action = self.select_action(state)
            self.t += 1

            # Update the target network
            # Global counter T is used in the original paper, but here we use
            # process specific counter instead. So i_target should be set
            # x-times smaller, where x is the number of processes
            if self.t % self.target_update_frequency == 0:
                self.logger.debug('sync')
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
