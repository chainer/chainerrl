from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()  # NOQA
from chainer import links as L
import numpy as np


def init_like_torch(link):
    # Mimic torch's default parameter initialization
    # TODO(muupan): Use chainer's initializers when it is merged
    for l in link.links():
        if isinstance(l, L.Linear):
            out_channels, in_channels = l.W.shape
            stdv = 1 / np.sqrt(in_channels)
            l.W.array[:] = np.random.uniform(-stdv, stdv, size=l.W.shape)
            if l.b is not None:
                l.b.array[:] = np.random.uniform(-stdv, stdv, size=l.b.shape)
        elif isinstance(l, L.Convolution2D):
            out_channels, in_channels, kh, kw = l.W.shape
            stdv = 1 / np.sqrt(in_channels * kh * kw)
            l.W.array[:] = np.random.uniform(-stdv, stdv, size=l.W.shape)
            if l.b is not None:
                l.b.array[:] = np.random.uniform(-stdv, stdv, size=l.b.shape)
