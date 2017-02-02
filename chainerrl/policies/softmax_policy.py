from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

from logging import getLogger
logger = getLogger(__name__)

import chainer

from chainerrl import distribution
from chainerrl.links.mlp import MLP
from chainerrl.policy import Policy


class SoftmaxPolicy(chainer.Chain, Policy):
    """Softmax policy that uses Boltzmann distributions.

    Args:
        model (chainer.Link):
            Link that is callable and outputs action values.
        beta (float):
            Parameter of Boltzmann distributions.
    """

    def __init__(self, model, beta=1.0):
        self.beta = beta
        super().__init__(model=model)

    def __call__(self, x, test=False):
        h = self.model(x, test=test)
        return distribution.SoftmaxDistribution(h, beta=self.beta)


class FCSoftmaxPolicy(SoftmaxPolicy):
    """Softmax policy that consists of FC layers and rectifiers"""

    def __init__(self, n_input_channels, n_actions,
                 n_hidden_layers=0, n_hidden_channels=None,
                 beta=1.0):
        self.n_input_channels = n_input_channels
        self.n_actions = n_actions
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels
        self.beta = beta

        super().__init__(
            model=MLP(n_input_channels,
                      n_actions,
                      (n_hidden_channels,) * n_hidden_layers),
            beta=self.beta)
