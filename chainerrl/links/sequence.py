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


class Sequence(chainer.ChainList, RecurrentChainMixin):

    def __init__(self, *layers):
        self.layers = layers
        links = [layer for layer in layers if isinstance(layer, chainer.Link)]
        # Cache the signatures because it might be slow
        self.argnames = [set(funcsigs.signature(layer).parameters)
                         for layer in layers]
        super().__init__(*links)

    def __call__(self, x, **kwargs):
        h = x
        for layer, argnames in zip(self.layers, self.argnames):
            layer_kwargs = {k: v for k, v in kwargs.items()
                            if k in argnames}
            h = layer(h, **layer_kwargs)
        return h
