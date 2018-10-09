from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

from chainerrl import explorer


class Greedy(explorer.Explorer):
    """No exploration"""

    def select_action(self, t, greedy_action_func, action_value=None):
        return greedy_action_func()

    def __repr__(self):
        return 'Greedy()'
