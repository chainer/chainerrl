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
from chainerrl.policy import Policy


class MellowmaxPolicy(chainer.Chain, Policy):
    """Mellowmax policy.

    See: http://arxiv.org/abs/1612.05628

    Args:
        model (chainer.Link):
            Link that is callable and outputs action values.
        omega (float):
            Parameter of the mellowmax function.
    """

    def __init__(self, model, omega=1.):
        self.omega = omega
        super().__init__(model=model)

    def __call__(self, x):
        h = self.model(x)
        return distribution.MellowmaxDistribution(h, omega=self.omega)
