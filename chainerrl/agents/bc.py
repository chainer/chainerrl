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


class BehavioralCloning(agent.AttributeSavingMixin, agent.ImitationAgent):
    """Behavioral Cloning Agent.

    Args:
        model :  Chainer model
        optimizer (Optimizer): Optimizer that is already setup
        explorer (Explorer): Explorer that specifies an exploration strategy.
        gpu (int): GPU device id if not None nor negative.
        minibatch_size (int): Minibatch size
        update_interval (int): Model update interval in step
        clip_delta (bool): Clip delta if set True
        phi (callable): Feature extractor applied to observations
        n_times_update (int): Number of repetition of update
        batch_accumulator (str): 'mean' or 'sum'
        logger (Logger): Logger used
        batch_states (callable): method which makes a batch of observations.
            default is `chainerrl.misc.batch_states.batch_states`
    """

    saved_attributes = ('model', 'optimizer')

    def __init__(self, model,
                 optimizer,
                 explorer, gpu=None,
                 minibatch_size=32, update_interval=1,
                 target_update_interval=10000, clip_delta=True,
                 phi=lambda x: x,
                 n_times_update=1,
                 batch_accumulator='mean',
                 logger=getLogger(__name__),
                 batch_states=batch_states):
        self.model = model

        if gpu is not None and gpu >= 0:
            cuda.get_device(gpu).use()
            self.model.to_gpu(device=gpu)

        self.xp = self.model.xp
        self.optimizer = optimizer
        self.explorer = explorer
        self.gpu = gpu
        self.clip_delta = clip_delta
        self.phi = phi
        self.batch_accumulator = batch_accumulator
        assert batch_accumulator in ('mean', 'sum')
        self.logger = logger
        self.batch_states = batch_states

        self.t = 0
        self.last_state = None
        self.last_action = None
        self.target_model = None


    def update(self, examples):
        """Update the model from examples

        Args:
            examples (list): List of tuples
                For behavioral cloning, each tuple must contain:
                  - observation (object): Observation
                  - action (object): Action
                  - reward (float): Reward
                  - next_observation (object): Next Observation
        Returns:
            None
        """
        exp_batch = batch_experiences(
            examples, xp=self.xp,
            phi=self.phi, gamma=self.gamma,
            batch_states=self.batch_states)
        if has_weight:
            exp_batch['weights'] = self.xp.asarray(
                [elem[0]['weight']for elem in experiences],
                dtype=self.xp.float32)
            if errors_out is None:
                errors_out = []
        loss = self._compute_loss(exp_batch, errors_out=errors_out)

        self.model.cleargrads()
        loss.backward()
        self.optimizer.update()


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

    def _compute_loss(self, exp_batch):
        """Compute the Behavioral Cloning loss for a batch of experiences


        Args:
          exp_batch (dict): A dict of batched arrays of transitions
        Returns:
          Computed loss from the minibatch of experiences
        """
        y, t = self._compute_y_and_t(exp_batch)

        return compute_value_loss(y, t, clip_delta=self.clip_delta,
                                  batch_accumulator=self.batch_accumulator)

    def act(self, obs):
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            action_value = self.model(
                self.batch_states([obs], self.xp, self.phi))
            q = float(action_value.max.array)
            action = cuda.to_cpu(action_value.greedy_actions.array)[0]


        self.logger.debug('t:%s q:%s action_value:%s', self.t, q, action_value)
        return action


    def stop_episode(self):
        self.last_state = None
        self.last_action = None
        if isinstance(self.model, Recurrent):
            self.model.reset_state()

    def get_statistics(self):
        pass