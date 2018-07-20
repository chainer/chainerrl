from chainer import initializer
from chainer.initializers import Constant
import numpy


class VarianceScalingConstant(initializer.Initializer):
    def __init__(self, scale=1.0, dtype=None):
        super(VarianceScalingConstant, self).__init__(dtype)
        self.scale = scale

    def __call__(self, array):
        if self.dtype is not None:
            assert array.dtype == self.dtype

        if len(array.shape) == 1:
            Constant(self.scale / numpy.sqrt(array.shape[0]))(array)
        else:
            fan_in, _ = initializer.get_fans(array.shape)

            Constant(self.scale / numpy.sqrt(fan_in))(array)
