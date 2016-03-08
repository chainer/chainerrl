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

