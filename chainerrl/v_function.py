from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from abc import ABCMeta
from abc import abstractmethod



class VFunction(object, metaclass=ABCMeta):

    @abstractmethod
    def __call__(self, x):
        raise NotImplementedError()
