from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

from abc import ABCMeta
from abc import abstractmethod

from future.utils import with_metaclass


class StateQFunction(with_metaclass(ABCMeta, object)):

    @abstractmethod
    def __call__(self, x):
        raise NotImplementedError()


class StateActionQFunction(with_metaclass(ABCMeta, object)):

    @abstractmethod
    def __call__(self, x, a):
        raise NotImplementedError()
