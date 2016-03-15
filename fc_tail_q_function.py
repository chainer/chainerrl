import numpy as np

import chainer
from chainer import functions as F
from chainer import links as L

import q_function
import dqn_net


class FCTailQFunction(chainer.ChainList, q_function.StateInputQFunction):

    def __init__(self, head, head_output_size, n_actions):

        self.n_actions = n_actions

        layers = [
            head.copy(),
            L.Linear(head_output_size, n_actions),
        ]

        super(FCTailQFunction, self).__init__(*layers)

    def forward(self, state, test=False):
        h = self[0](state)
        return self[1](h)
