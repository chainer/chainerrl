from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()
import chainer.functions as F
# from chainer.optimizer import WeightDecay

import copy
import numpy as np

from chainerrl.agents.dqn import DQN
from chainerrl.functions import scale_grad


class ResidualCorrection(object):
    def __init__(self, agent):
        self.agent = agent
        self.name = 'ResidualCorrection'

    def __call__(self, opt):
        gdirected = [param.grad for param in self.agent.q_function.params()]
        gresidual = [g + param.grad for g, param in
                     zip(gdirected, self.agent.q_function_copy.params())]
        inner_rd = sum([np.dot(gr.flatten(), gd.flatten())
                        for gr, gd in zip(gresidual, gdirected)])
        if inner_rd >= 0.0:
            # print((inner_rd, None, None))
            return
        inner_rr = sum([np.dot(gr.flatten(), gr.flatten())
                        for gr in gresidual])
        phi = - inner_rd / (inner_rr - inner_rd)
        # print((inner_rd, inner_rr, phi))
        gfix = [phi * (gr - gd) for gr, gd in zip(gresidual, gdirected)]
        for p, g in zip(self.agent.q_function.params(), gfix):
            p.grad += g


class ResidualDQN(DQN):
    """DQN that allows maxQ also backpropagate gradients.

    Args:
        grad_scale (float or None): Scale of gradient of maxQ.
            Constant scale (residual graident algorithm) if a float is given.
            Residual algorithm if None is given.
    """

    def __init__(self, *args, **kwargs):
        self.grad_scale = kwargs.pop('grad_scale', None)
        if self.grad_scale is not None:
            assert(0.0 <= self.grad_scale <= 1.0)
        super().__init__(*args, **kwargs)
        if self.grad_scale is None:
            self.optimizer.add_hook(ResidualCorrection(self))

    def sync_target_network(self):
        pass

    def _compute_target_values(self, exp_batch, gamma):

        batch_next_state = exp_batch['next_state']

        if self.grad_scale is None:
            self.q_function_copy = copy.deepcopy(self.q_function)
            self.q_function_copy.cleargrads()
            target_next_qout = self.q_function_copy(
                batch_next_state, test=False)
        else:
            target_next_qout = self.q_function(batch_next_state, test=False)
        next_q_max = target_next_qout.max

        batch_rewards = exp_batch['reward']
        batch_terminal = exp_batch['is_state_terminal']

        return batch_rewards + self.gamma * (1.0 - batch_terminal) * next_q_max

    def _compute_y_and_t(self, exp_batch, gamma):

        batch_state = exp_batch['state']
        batch_size = len(batch_state)

        # Compute Q-values for current states
        qout = self.q_function(batch_state, test=False)

        batch_actions = exp_batch['action']
        batch_q = F.reshape(qout.evaluate_actions(
            batch_actions), (batch_size, 1))

        # Target values must also backprop gradients
        batch_q_target = F.reshape(
            self._compute_target_values(exp_batch, gamma), (batch_size, 1))

        if self.grad_scale is not None:
            batch_q_target = scale_grad.scale_grad(
                batch_q_target, self.grad_scale)

        return batch_q, batch_q_target

    @property
    def saved_attributes(self):
        # ResidualDQN doesn't use target models
        return ('model', 'optimizer')

    def input_initial_batch_to_target_model(self, batch):
        pass
