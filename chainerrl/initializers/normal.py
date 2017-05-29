import chainer
import numpy as np


class LeCunNormal(chainer.initializers.HeNormal):
    """sorry

    """
    def __init__(self, scale=1.0, dtype=None):
        super(LeCunNormal, self).__init__(np.sqrt(0.5)*scale, dtype)
