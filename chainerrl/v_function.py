from abc import ABCMeta
from abc import abstractmethod


class VFunction(object, metaclass=ABCMeta):

    @abstractmethod
    def __call__(self, x):
        raise NotImplementedError()
