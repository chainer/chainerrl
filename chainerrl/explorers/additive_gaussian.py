from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

import numpy as np

from chainerrl import explorer


class AdditiveGaussian(explorer.Explorer):
    """Additive Gaussian noise"""

    def __init__(self, scale):
        self.scale = scale

    def select_action(self, t, greedy_action_func, action_value=None):
        a = greedy_action_func()
        noise = np.random.normal(
            scale=self.scale, size=a.shape).astype(np.float32)
        return a + noise

    def __repr__(self):
        return 'AdditiveGaussian(scale={})'.format(self.scale)
