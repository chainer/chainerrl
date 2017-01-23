from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import super
from future import standard_library
standard_library.install_aliases()
import inspect

import chainer

from chainerrl.recurrent import RecurrentChainMixin


class Sequence(chainer.ChainList, RecurrentChainMixin):

    def __init__(self, *layers):
        self.layers = layers
        links = [layer for layer in layers if isinstance(layer, chainer.Link)]
        super().__init__(*links)

    def __call__(self, x, **kwargs):
        h = x
        for layer in self.layers:
            layer_argnames = inspect.getargspec(layer)[0]
            layer_kwargs = {k: v for k, v in kwargs.items()
                            if k in layer_argnames}
            h = layer(h, **layer_kwargs)
        return h
