from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
from future.utils import with_metaclass
standard_library.install_aliases()

from abc import ABCMeta
from abc import abstractmethod
import collections
import contextlib

import chainer


def unchain_backward(state):
    """Call Variable.unchain_backward recursively."""
    if isinstance(state, collections.Iterable):
        for s in state:
            unchain_backward(s)
    elif isinstance(state, chainer.Variable):
        state.unchain_backward()


class Recurrent(with_metaclass(ABCMeta, object)):
    """Interface of recurrent models."""

    __state_stack = []

    @abstractmethod
    def get_state(self):
        raise NotImplementedError()

    @abstractmethod
    def set_state(self, state):
        raise NotImplementedError()

    @abstractmethod
    def reset_state(self):
        raise NotImplementedError()

    def unchain_backward(self):
        unchain_backward(self.get_state())

    def push_state(self):
        self.__state_stack.append(self.get_state())
        self.reset_state()

    def pop_state(self):
        self.set_state(self.__state_stack.pop())

    def push_and_keep_state(self):
        self.__state_stack.append(self.get_state())

    def update_state(self, *args, **kwargs):
        """Update its state as if self.__call__ is called.

        Unlike __call__, stateless objects may do nothing.
        """
        self(*args, **kwargs)

    @contextlib.contextmanager
    def state_reset(self):
        self.push_state()
        yield
        self.pop_state()

    @contextlib.contextmanager
    def state_kept(self):
        self.push_and_keep_state()
        yield
        self.pop_state()


def get_state(chain):
    assert isinstance(chain, (chainer.Chain, chainer.ChainList))
    state = []
    for l in chain.children():
        if isinstance(l, chainer.links.LSTM):
            state.append((l.c, l.h))
        elif isinstance(l, Recurrent):
            state.append(l.get_state())
        elif isinstance(l, (chainer.Chain, chainer.ChainList)):
            state.append(get_state(l))
        else:
            state.append(None)
    return state


def stateful_links(chain):
    for l in chain.children():
        if isinstance(l, (chainer.links.LSTM, Recurrent)):
            yield l
        elif isinstance(l, (chainer.Chain, chainer.ChainList)):
            for m in stateful_links(l):
                yield m


def set_state(chain, state):
    assert isinstance(chain, (chainer.Chain, chainer.ChainList))
    for l, s in zip(chain.children(), state):
        if isinstance(l, chainer.links.LSTM):
            c, h = s
            # LSTM.set_state doesn't accept None state
            if c is not None:
                l.set_state(c, h)
        elif isinstance(l, Recurrent):
            l.set_state(s)
        elif isinstance(l, (chainer.Chain, chainer.ChainList)):
            set_state(l, s)
        else:
            assert s is None


def reset_state(chain):
    assert isinstance(chain, (chainer.Chain, chainer.ChainList))
    for l in chain.children():
        if isinstance(l, chainer.links.LSTM):
            l.reset_state()
        elif isinstance(l, Recurrent):
            l.reset_state()
        elif isinstance(l, (chainer.Chain, chainer.ChainList)):
            reset_state(l)


class RecurrentChainMixin(Recurrent):
    """Mixin that aggregate states of children.

    This mixin can only applied to chainer.Chain or chainer.ChainLink. The
    resulting class will implement Recurrent by searching recurrent models
    recursively from its children.
    """

    def get_state(self):
        return get_state(self)

    def set_state(self, state):
        set_state(self, state)

    def reset_state(self):
        reset_state(self)


@contextlib.contextmanager
def state_kept(link):
    if isinstance(link, Recurrent):
        link.push_and_keep_state()
        yield
        link.pop_state()
    else:
        yield


@contextlib.contextmanager
def state_reset(link):
    if isinstance(link, Recurrent):
        link.push_state()
        yield
        link.pop_state()
    else:
        yield
