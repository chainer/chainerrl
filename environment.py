from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
class Environment(object):
    """RL learning environment
    """

    @property
    def state(self):
        pass

    @property
    def reward(self):
        pass

    def receive_action(self, action):
        pass

class EpisodicEnvironment(Environment):

    def initialize(self):
        """
        Initialize the internal state
        """
        pass

    @property
    def is_terminal(self):
        pass

