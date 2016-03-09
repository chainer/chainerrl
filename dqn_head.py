import numpy as np

import chainer
from chainer import functions as F
from chainer import links as L


class NatureDQNHead(chainer.ChainList):

    def __init__(self, n_input_channels=4):
        self.n_input_channels = n_input_channels

        layers = [
            L.Convolution2D(n_input_channels, 32, 8, stride=4),
            L.Convolution2D(32, 64, 4, stride=2),
            L.Convolution2D(64, 64, 3, stride=1),
            L.Linear(3136, 512),
        ]

        super(NatureDQNHead, self).__init__(*layers)

    def __call__(self, state):
        h = chainer.Variable(state)
        for layer in self:
            h = F.relu(layer(h))
        return h

class NIPSDQNHead(chainer.ChainList):

    def __init__(self, n_input_channels=4):
        self.n_input_channels = n_input_channels

        layers = [
            L.Convolution2D(n_input_channels, 16, 8, stride=4),
            L.Convolution2D(16, 32, 4, stride=2),
            L.Linear(2592, 256),
        ]

        super(NIPSDQNHead, self).__init__(*layers)

    def __call__(self, state):
        h = chainer.Variable(state)
        for layer in self:
            h = F.relu(layer(h))
        return h
