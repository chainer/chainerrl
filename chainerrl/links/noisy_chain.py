import chainer
from chainer.links import Linear

from chainerrl.initializers import LeCunNormal
from chainerrl.links.noisy_linear import FactorizedNoisyLinear


def to_factorized_noisy(link):
    _map_links(_func_to_factorized_noisy, link)


def _func_to_factorized_noisy(link):
    if isinstance(link, Linear):
        return FactorizedNoisyLinear(link)
    else:
        return link


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
                    setattr(link, name, func(child))
    elif isinstance(link, chainer.ChainList):
        children = link._children
        for i in range(len(children)):
            child = children[i]
            new_child = func(child)
            if new_child is child:
                _map_links(func, child)
            else:
                # mimic ChainList.add_link
                children[i] = func(child)
                children[i].name = str(i)
