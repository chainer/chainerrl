from chainer import links as L
import numpy as np


def init_like_torch(link):
    # Mimic torch's default parameter initialization
    # TODO(muupan): Use chainer's initializers when it is merged
    for li in link.links():
        if isinstance(li, L.Linear):
            out_channels, in_channels = li.W.shape
            stdv = 1 / np.sqrt(in_channels)
            li.W.array[:] = np.random.uniform(-stdv, stdv, size=li.W.shape)
            if li.b is not None:
                li.b.array[:] = np.random.uniform(-stdv, stdv, size=li.b.shape)
        elif isinstance(li, L.Convolution2D):
            out_channels, in_channels, kh, kw = li.W.shape
            stdv = 1 / np.sqrt(in_channels * kh * kw)
            li.W.array[:] = np.random.uniform(-stdv, stdv, size=li.W.shape)
            if li.b is not None:
                li.b.array[:] = np.random.uniform(-stdv, stdv, size=li.b.shape)
