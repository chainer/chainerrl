from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import

from abc import ABCMeta
from abc import abstractmethod


class Explorer(object, metaclass=ABCMeta):
    """Abstract explorer."""

    @abstractmethod
    def select_action(self, t, greedy_action_func, action_value=None):
        """Select an action.

        Args:
          t: current time step
          greedy_action_func: function with no argument that returns an action
          action_value (ActionValue): ActionValue object
        """
        raise NotImplementedError()
