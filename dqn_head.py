import chainer
from chainer import functions as F
from chainer import links as L

from wn_convolution_2d import WNConvolution2D
from wn_linear import WNLinear
import crelu


class NatureDQNHead(chainer.ChainList):

    def __init__(self, n_input_channels=4, n_output_channels=512,
                 activation=F.relu, bias=0.1):
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.n_output_channels = n_output_channels

        layers = [
            L.Convolution2D(n_input_channels, 32, 8, stride=4, bias=bias),
            L.Convolution2D(32, 64, 4, stride=2, bias=bias),
            L.Convolution2D(64, 64, 3, stride=1, bias=bias),
            L.Linear(3136, n_output_channels, bias=bias),
        ]

        super(NatureDQNHead, self).__init__(*layers)

    def __call__(self, state):
        h = chainer.Variable(state)
        for layer in self:
            h = self.activation(layer(h))
        return h


class NIPSDQNHead(chainer.ChainList):

    def __init__(self, n_input_channels=4, n_output_channels=256,
                 activation=F.relu, bias=0.1):
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.n_output_channels = n_output_channels

        layers = [
            L.Convolution2D(n_input_channels, 16, 8, stride=4, bias=bias),
            L.Convolution2D(16, 32, 4, stride=2, bias=bias),
            L.Linear(2592, n_output_channels, bias=bias),
        ]

        super(NIPSDQNHead, self).__init__(*layers)

    def __call__(self, state):
        h = chainer.Variable(state)
        for layer in self:
            h = self.activation(layer(h))
        return h


class NatureDQNHeadCReLU(chainer.ChainList):

    def __init__(self, n_input_channels=4, n_output_channels=512,
                 activation=F.relu, bias=0.1):
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.n_output_channels = n_output_channels

        layers = [
            L.Convolution2D(n_input_channels, 32 / 2, 8, stride=4, bias=bias),
            L.Convolution2D(32, 64 / 2, 4, stride=2, bias=bias),
            L.Convolution2D(64, 64 / 2, 3, stride=1, bias=bias),
            L.Linear(3136, n_output_channels, bias=bias),
        ]

        super(NatureDQNHead, self).__init__(*layers)

    def __call__(self, state):
        h = chainer.Variable(state)
        for layer in self:
            h = self.activation(layer(h))
        return h


class NIPSDQNHeadCReLU(chainer.ChainList):

    def __init__(self, n_input_channels=4, n_output_channels=256):
        self.n_input_channels = n_input_channels
        self.n_output_channels = n_output_channels

        layers = [
            L.Convolution2D(n_input_channels, 16 / 2, 8, stride=4),
            L.Convolution2D(16, 32 / 2, 4, stride=2),
            L.Linear(2592, n_output_channels / 2),
        ]

        super(NIPSDQNHeadCReLU, self).__init__(*layers)

    def __call__(self, state):
        h = chainer.Variable(state)
        for layer in self:
            h = crelu.crelu(layer(h))
        return h


class WNNIPSDQNHead(chainer.ChainList):

    def __init__(self, n_input_channels=4, n_output_channels=256,
                 activation=F.relu, bias=0.1):
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.n_output_channels = n_output_channels

        layers = [
            WNConvolution2D(n_input_channels, 16, 8, stride=4, bias=bias),
            WNConvolution2D(16, 32, 4, stride=2, bias=bias),
            WNLinear(2592, n_output_channels, bias=bias),
        ]

        super(WNNIPSDQNHead, self).__init__(*layers)

    def __call__(self, state):
        h = chainer.Variable(state)
        for layer in self:
            h = self.activation(layer(h))
        return h
