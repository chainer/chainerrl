import numpy as np

import chainer
from chainer import functions as F
from chainer import links as L

import v_function
import dqn_net


class DQNVFunction(chainer.ChainList, v_function.VFunction):

    def __init__(self, n_input_channels=4, input_w=84, input_h=84):

        layers = [
            dqn_net.DQNNet(
                n_input_channels=n_input_channels,
                input_w=input_w, input_h=input_h,
                n_output=1
            ),
        ]

        super(DQNVFunction, self).__init__(*layers)

    def __call__(self, state):
        return self[0](state)
