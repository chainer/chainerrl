from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import super
from future import standard_library
standard_library.install_aliases()
import chainer
from chainer import functions as F

from links.wn_convolution_2d import WNConvolution2D
from links.wn_linear import WNLinear


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
        h = state
        for layer in self:
            h = self.activation(layer(h))
        return h
