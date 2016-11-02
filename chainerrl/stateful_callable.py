from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from builtins import super
from builtins import range
from future import standard_library
from future.utils import with_metaclass
standard_library.install_aliases()

from abc import ABCMeta


class StatefulCallable(with_metaclass(ABCMeta, object)):

    def push_state(self):
        pass

    def pop_state(self):
        pass

    def reset_state(self):
        pass

    def push_and_keep_state(self):
        pass

    def update_state(self, *args, **kwargs):
        """Update its state as if self.__call__ is called.

        Unlike __call__, stateless callables may do nothing.
        """
        pass

    def unchain_backward(self):
        pass
