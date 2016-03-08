import numpy as np

import chainer
from chainer import functions as F
from chainer import links as L


class DQNNet(chainer.ChainList):

    def __init__(self, n_input_channels=4, input_w=84, input_h=84,
                 n_output=18):
        self.n_input_channels = n_input_channels
        self.input_w = input_w
        self.input_h = input_h
        self.n_output = n_output

        layers = [
            L.Convolution2D(n_input_channels, 32, 8, stride=4),
            L.Convolution2D(32, 64, 4, stride=2),
            L.Convolution2D(64, 64, 3, stride=1),
            L.Linear(3136, 512),
            L.Linear(512, n_output),
        ]

        super(DQNNet, self).__init__(*layers)

    def __call__(self, state):
        h = chainer.Variable(state)
        for layer in self[:-1]:
            h = F.relu(layer(h))
        h = self[-1](h)
        return h
