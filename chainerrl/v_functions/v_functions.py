from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

import chainer

from chainerrl.links.mlp import MLP
from chainerrl.recurrent import RecurrentChainMixin
from chainerrl.v_function import VFunction


class SingleModelVFunction(
        chainer.Chain, VFunction, RecurrentChainMixin):
    """Q-function with discrete actions.

    Args:
        model (chainer.Link):
            Link that is callable and outputs action values.
    """

    def __init__(self, model):
        super().__init__(model=model)

    def __call__(self, x, test=False):
        h = self.model(x, test=test)
        return h


class FCVFunction(SingleModelVFunction):

    def __init__(self, n_input_channels, n_hidden_layers=0,
                 n_hidden_channels=None):
        self.n_input_channels = n_input_channels
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels

        super().__init__(
            model=MLP(self.n_input_channels, 1,
                      [self.n_hidden_channels] * self.n_hidden_layers),
        )
