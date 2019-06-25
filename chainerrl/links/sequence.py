from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import chainer

from chainerrl.recurrent import RecurrentChainMixin

try:
    # For Python 3.5 and later
    from inspect import Parameter
    from inspect import signature
except Exception:
    from funcsigs import Parameter
    from funcsigs import signature


def accept_variable_arguments(func):
    for param in signature(func).parameters.values():
        if param.kind in (Parameter.VAR_POSITIONAL,
                          Parameter.VAR_KEYWORD):
            return True
    return False


class Sequence(chainer.ChainList, RecurrentChainMixin):
    """Sequential callable Link that consists of other Links."""

    def __init__(self, *layers):
        self.layers = list(layers)
        self.linkidx_to_layer = dict()
        links = [layer for layer in layers if isinstance(layer, chainer.Link)]
        # Cache the signatures because it might be slow
        self.argnames = [set(signature(layer).parameters)
                         for layer in layers]
        self.accept_var_args = [accept_variable_arguments(layer)
                                for layer in layers]
        super().__init__(*links)
        dummy_links = []
        for i in range(len(layers)):
            layer = layers[i]
            if isinstance(layer, chainer.Link):
                # needed for noisy net compatibility
                assert layer in self._children
                dummy_links.append(layer)
                self.linkidx_to_layer[len(dummy_links) - 1] = i 
        assert dummy_links == links
        assert self._children == dummy_links

    def __call__(self, x, **kwargs):
        h = x
        for layer, argnames, accept_var_args in zip(self.layers,
                                                    self.argnames,
                                                    self.accept_var_args):
            if accept_var_args:
                layer_kwargs = kwargs
            else:
                layer_kwargs = {k: v for k, v in kwargs.items()
                                if k in argnames}
            h = layer(h, **layer_kwargs)
        return h
