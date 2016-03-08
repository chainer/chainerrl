import numpy as np

import chainer
from chainer import functions as F
from chainer import links as L

import policy
import dqn_net


class DQNPolicy(chainer.ChainList, policy.SoftmaxPolicy):

    def __init__(self, n_input_channels=4, input_w=84, input_h=84,
                 n_actions=18):
        layers = [
            dqn_net.DQNNet(
                n_input_channels=n_input_channels,
                input_w=input_w, input_h=input_h,
                n_output=n_actions
            ),
        ]

        super(DQNPolicy, self).__init__(*layers)

    def forward(self, state):
        return self[0](state)
