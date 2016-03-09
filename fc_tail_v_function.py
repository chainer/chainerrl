import numpy as np

import chainer
from chainer import functions as F
from chainer import links as L

import v_function
import dqn_net


class FCTailVFunction(chainer.ChainList, v_function.VFunction):

    def __init__(self, head, head_output_size):

        layers = [
            head.copy(),
            L.Linear(head_output_size, 1),
        ]

        super(FCTailVFunction, self).__init__(*layers)

    def __call__(self, state):
        h = self[0](state)
        return self[1](h)
