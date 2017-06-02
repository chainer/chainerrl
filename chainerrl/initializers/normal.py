import chainer
import numpy as np


class LeCunNormal(chainer.initializers.HeNormal):
    """LeCunNormal is (essentially) the default initializer in Chainer v1.

    chainer.initializers.LeCunNormal is not available yet.
    (Chainer Pull Request #2764 has not been merged.)
    """
    def __init__(self, scale=1.0, dtype=None):
        super(LeCunNormal, self).__init__(np.sqrt(0.5)*scale, dtype)
