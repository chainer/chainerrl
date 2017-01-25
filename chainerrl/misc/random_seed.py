from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

import os
import random

import numpy as np


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['CHAINER_SEED'] = str(seed)
