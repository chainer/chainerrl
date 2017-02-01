from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

import chainer
import funcsigs

from chainerrl.recurrent import RecurrentChainMixin


def accept_variable_arguments(func):
    for param in funcsigs.signature(func).parameters.values():
        if param.kind in (funcsigs.Parameter.VAR_POSITIONAL,
                          funcsigs.Parameter.VAR_KEYWORD):
            return True
    return False


class Sequence(chainer.ChainList, RecurrentChainMixin):
    """Sequential callable Link that consists of other Links."""

    def __init__(self, *layers):
        self.layers = layers
        links = [layer for layer in layers if isinstance(layer, chainer.Link)]
        # Cache the signatures because it might be slow
        self.argnames = [set(funcsigs.signature(layer).parameters)
                         for layer in layers]
        self.accept_var_args = [accept_variable_arguments(layer)
                                for layer in layers]
        super().__init__(*links)

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
