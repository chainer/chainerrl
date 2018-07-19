from chainer import initializer
from chainer.initializers import Uniform
import numpy


class VarianceScalingUniform(initializer.Initializer):
    def __init__(self, scale=1.0, dtype=None):
        super(VarianceScalingUniform, self).__init__(dtype)
        self.scale = scale

    def __call__(self, array):
        if self.dtype is not None:
            assert array.dtype == self.dtype

        if len(array.shape) == 1:
            Uniform(self.scale / numpy.sqrt(array.shape[0]))(array)
        else:
            fan_in, _ = initializer.get_fans(array.shape)
            Uniform(self.scale / numpy.sqrt(fan_in))(array)
