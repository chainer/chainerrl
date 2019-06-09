from abc import ABCMeta
from abc import abstractmethod



class StateQFunction(object, metaclass=ABCMeta):

    @abstractmethod
    def __call__(self, x):
        raise NotImplementedError()


class StateActionQFunction(object, metaclass=ABCMeta):

    @abstractmethod
    def __call__(self, x, a):
        raise NotImplementedError()
