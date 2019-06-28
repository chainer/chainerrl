"""Noisy Networks

See http://arxiv.org/abs/1706.10295
"""

import chainer
from chainer.links import Linear

from chainerrl.links.noisy_linear import FactorizedNoisyLinear
from chainerrl.links.sequence import Sequence


def to_factorized_noisy(link, *args, **kwargs):
    """Add noisiness to components of given link

    Currently this function supports L.Linear (with and without bias)
    """

    def func_to_factorized_noisy(link):
        if isinstance(link, Linear):
            return FactorizedNoisyLinear(link, *args, **kwargs)
        else:
            return link

    _map_links(func_to_factorized_noisy, link)


def _map_links(func, link):
    if isinstance(link, chainer.Chain):
        children_names = link._children.copy()
        for name in children_names:
            child = getattr(link, name)
            new_child = func(child)
            if new_child is child:
                _map_links(func, child)
            else:
                delattr(link, name)
                with link.init_scope():
                    setattr(link, name, new_child)
    elif isinstance(link, chainer.ChainList):
        children = link._children
        for i in range(len(children)):
            child = children[i]
            new_child = func(child)
            if new_child is child:
                _map_links(func, child)
            else:
                # mimic ChainList.add_link
                children[i] = new_child
                children[i].name = str(i)

                if isinstance(link, Sequence):
                    _replace_unique_item(link.layers, child, new_child)
                # Check chainer.Sequential if it exists.
                sequential_class = getattr(chainer, 'Sequential', ())
                if isinstance(link, sequential_class):
                    _replace_unique_item(link._layers, child, new_child)


def _replace_unique_item(xs, old, new):
    indices = [i for i, x in enumerate(xs) if x is old]
    assert len(indices) == 1
    xs[indices[0]] = new
