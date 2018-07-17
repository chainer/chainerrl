"""Noisy Networks

See http://arxiv.org/abs/1706.10295
"""

import chainer
from chainer.links import Linear

from chainerrl.links.noisy_linear import FactorizedNoisyLinear
from chainerrl.links.noisy_linear2 import FactorizedNoisyLinear2
from logging import getLogger
from chainerrl import links

def to_factorized_noisy2(link, *args, **kwargs):
    """Add noisiness to components of given link

    Currently this function supports L.Linear (with and without bias)
    """
    links = []

    def func_to_factorized_noisy(link):
        if isinstance(link, Linear):
            a = FactorizedNoisyLinear2(link, *args, **kwargs)
            links.append(a)
            return a
        else:
            return link

    _map_links(func_to_factorized_noisy, link)
    return links

def to_factorized_noisy(link, *args, **kwargs):
    """Add noisiness to components of given link

    Currently this function supports L.Linear (with and without bias)
    """
    links = []

    def func_to_factorized_noisy(link):
        if isinstance(link, Linear):
            a = FactorizedNoisyLinear(link, *args, **kwargs)
            links.append(a)
            return a
        else:
            return link

    _map_links(func_to_factorized_noisy, link)
    return links

def _map_links(func, link):
    logger = getLogger(__name__)

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
                    links.append(new_child)
    elif isinstance(link, chainer.ChainList):
        children = link._children
        logger.info(children)
        for i in range(len(children)):
            child = children[i]
            new_child = func(child)
            if new_child is child:
                _map_links(func, child)
            else:
                # mimic ChainList.add_link
                # logger.info("replace {}, {}".format(child.W.shape, new_child))
                children[i] = new_child
                children[i].name = str(i)

                if isinstance(link, links.Sequence):
                    link.layers[i] = new_child
