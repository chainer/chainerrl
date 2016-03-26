import chainer
from chainer import functions as F
from chainer import links as L


class NatureDQNHead(chainer.ChainList):

    def __init__(self, n_input_channels=4, activation=F.relu, bias=0.1):
        self.n_input_channels = n_input_channels
        self.activation = activation

        layers = [
            L.Convolution2D(n_input_channels, 32, 8, stride=4, bias=bias),
            L.Convolution2D(32, 64, 4, stride=2, bias=bias),
            L.Convolution2D(64, 64, 3, stride=1, bias=bias),
            L.Linear(3136, 512, bias=bias),
        ]

        super(NatureDQNHead, self).__init__(*layers)

    def __call__(self, state):
        h = chainer.Variable(state)
        for layer in self:
            h = self.activation(layer(h))
        return h


class NIPSDQNHead(chainer.ChainList):

    def __init__(self, n_input_channels=4, activation=F.relu):
        self.n_input_channels = n_input_channels
        self.activation = activation

        layers = [
            L.Convolution2D(n_input_channels, 16, 8, stride=4, bias=0.1),
            L.Convolution2D(16, 32, 4, stride=2, bias=0.1),
            L.Linear(2592, 256, bias=0.1),
        ]

        super(NIPSDQNHead, self).__init__(*layers)

    def __call__(self, state):
        h = chainer.Variable(state)
        for layer in self:
            h = self.activation(layer(h))
        return h
