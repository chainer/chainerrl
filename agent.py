class Agent(object):
    """Abstract agent class.
    """

    def act(self, state, reward, **kwargs):
        """
        Returns:
          ~object: action
        """
        raise NotImplementedError
