from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
class Agent(object):
    """Abstract agent class.
    """

    def act(self, state, reward, **kwargs):
        """
        Returns:
          ~object: action
        """
        raise NotImplementedError
