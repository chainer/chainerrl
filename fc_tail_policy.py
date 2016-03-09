import numpy as np

import chainer
from chainer import functions as F
from chainer import links as L

import policy
import dqn_net


class FCTailPolicy(chainer.ChainList, policy.SoftmaxPolicy):

    def __init__(self, head, head_output_size, n_actions=18):
        layers = [
            head.copy(),
            L.Linear(head_output_size, n_actions),
        ]
        super(FCTailPolicy, self).__init__(*layers)

    def forward(self, state):
        h = self[0](state)
        return self[1](h)
