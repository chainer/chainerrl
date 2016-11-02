from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import range
from future import standard_library
standard_library.install_aliases()
import unittest
import random

import chainer
import numpy as np

from chainerrl import q_function


class TestQFunction(unittest.TestCase):

    def setUp(self):
        pass
